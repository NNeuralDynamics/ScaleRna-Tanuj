#!/usr/bin/env python
"""
Filter barcodes to call cells and generate filtered expression matrix. 
"""

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import sys
from typing import Optional

import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy.stats import median_abs_deviation
from scipy import sparse

from cellFinder import construct_ambient_profile, test_ambiguous_barcodes 
from scale_utils import io
from scale_utils.stats import extrapolate_unique


@dataclass(frozen=True)
class CellCallingOptions:
    """
    This class holds options for cell calling.

    Attributes:
        fixedCells: If true the top fixedCells ranked by UTC are called as cells.
        expectedCells: Expected number of cells to call.
        topCellPercent: Percentile of cells sorted descending by UMI counts to help calculate the unique transcript counts threshold.
        minCellRatio: Minimum ratio of unique transcript counts to top cell UMI counts to help calculate the unique transcript counts threshold.
        minUTC: The minimum number of unique transcript counts a barcode must be associated with to be considered a potential cell.
        UTC: The minimum number of unique transcript counts a barcode must be associated with to be considered a called cell.
        cellFinder: If true Cell Finder will be used for cell calling.
        FDR: False discovery rate to use for rescuing cells based on deviation from ambient profile.
        alpha: Overdispersion parameter for Dirichlet Multinomial distribution. If none, estimated from ambient barcodes.
    """
    fixedCells: bool
    expectedCells: int
    topCellPercent: int
    minCellRatio: float
    minUTC: int
    UTC: int
    cellFinder: bool
    FDR: float
    alpha: Optional[float]

@dataclass(frozen=True)
class OutlierOptions:
    """
    Options for flagging / filtering cell outliers based on QC metrics
    """
    filter_outliers: bool
    reads_mads: float
    passing_mads: float
    mito_mads: float
    mito_min_thres: float = 0.05

def classify_barcodes(total_counts: np.ndarray, options: CellCallingOptions) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Classify cell barcodes into ambient, cell and (optionally) ambiguous,
    based on total transcript counts per barcode

    Args:
        total_counts: Total unique transcript counts for all barcodes
    
    Returns:
        An array of indices for ambient barcodes
        An array of indices for ambiguous barcodes, if CellFinder is enabled
        An array of indices for top-cell barcodes
    """
    ambiguous_barcode_indices = np.empty(0)
    ambient_barcode_indices = np.where(total_counts < options.minUTC)[0]
    threshold = options.UTC or calculate_UTC_threshold(total_counts, ambient_barcode_indices, options=options)

    if options.cellFinder:
        cell_barcode_indices = np.where(total_counts >= threshold)[0]
        ambiguous_barcode_indices = np.setdiff1d(np.setdiff1d(np.arange(len(total_counts)), ambient_barcode_indices), cell_barcode_indices)
    elif options.fixedCells:
        if options.expectedCells < 0:
            raise ValueError('--expectedCells must be a positive integer.')
        cell_barcode_indices = np.argsort(total_counts)[::-1][:options.expectedCells]
        ambient_barcode_indices = np.setdiff1d(np.arange(len(total_counts)), cell_barcode_indices)
    else: #'Top-cells" UTC Thresholding
        cell_barcode_indices = np.where(total_counts >= threshold)[0]
        ambient_barcode_indices = np.setdiff1d(np.arange(len(total_counts)), cell_barcode_indices)
    return ambient_barcode_indices, ambiguous_barcode_indices, cell_barcode_indices

def calculate_UTC_threshold(total_counts: np.ndarray, ambient_barcode_indices: np.ndarray, options: CellCallingOptions) -> int:
    """
    Calculate the unique transcripts counts threshold above which all barcodes are called as cells

    Args:
        total_counts: Total unique transcript counts for all barcodes
        ambient_barcode_indices: Array of indices for barcodes below minUTC
        options: Options for cell calling

    Returns: 
        The unique transcript counts threshold
    """
    
    if options.topCellPercent < 1 or options.topCellPercent > 99:
        raise ValueError('--topCellPercent must fall in the range 1 <= x <= 99.')
    if options.minCellRatio < 1:
        raise ValueError('--minCellRatio must be >= 1.')     
    if options.expectedCells > 0 and options.expectedCells < len(total_counts):
        # when --expectedCells is set, use the top --expectedCells barcodes as preliminary cells
        preliminary_cell_unique_transcript_counts = total_counts[np.argsort(total_counts)[::-1]][:options.expectedCells]
    else:
        preliminary_cell_unique_transcript_counts = total_counts[np.setdiff1d(np.arange(len(total_counts)), ambient_barcode_indices)]

    # Calculate unique transcripts counts threshold
    threshold = 0
    if len(preliminary_cell_unique_transcript_counts) != 0:
        # Calculate the UTC threshold as the UMI count associated with the --topCellPercent of the preliminary cells divided by --minCellRatio.
        threshold = np.round(np.percentile(preliminary_cell_unique_transcript_counts, options.topCellPercent) / options.minCellRatio)
    
    # Don't return threshold below minUTC
    return max(threshold, options.minUTC)

def call_cell_barcodes(mtx: Path, sampleMetrics: Path, options: CellCallingOptions) -> pd.DataFrame:
    """
    Call cells in this sample using thresholding options, including optionally
    rescuing ambiguous barcodes as cells with the CellFinder algorithm

    Args:
        mtx: Cell-by-gene count matrix file in .mtx format
        sampleMetrics: allCells.csv containing metrics computed across all barcodes in this sample
        options: Options for cell calling
        
    Returns:
        The allCells.csv for this sample, with updated columns to show cell pass/fail and optional flags
    """
    all_cells = pd.read_csv(sampleMetrics, index_col=0)
    mtx = sparse.csc_array(mmread(mtx))
    total_counts = all_cells['umis'].to_numpy()
    # Classify cells based on 'topCell' threshold
    # If CellFinder is enabled, 'ambiguous' are the barcodes to test
    ambient_bcs, ambiguous_bcs, cell_bcs = classify_barcodes(total_counts, options=options)
    flags = pd.Series('', index=all_cells.index)
    passed = pd.Series(False, index=all_cells.index)
    passed.iloc[cell_bcs] = True
    if len(ambiguous_bcs) > 0: # Run CellFinder
        # Make gene counts discrete so Good-Turing algorithm can work
        discrete_mtx = mtx[:]
        discrete_mtx.data = np.round(discrete_mtx.data)
        # Disregard genes with zero counts
        discrete_mtx = discrete_mtx[discrete_mtx.sum(axis=1) > 0, :]
        # Test each ambiguous barcode against the ambient profile
        ambient_profile = construct_ambient_profile(discrete_mtx, ambient_bcs)
        FDRs = test_ambiguous_barcodes(discrete_mtx, ambient_profile, ambient_bcs, ambiguous_bcs, alpha=options.alpha)
        passed.iloc[ambiguous_bcs] = FDRs <= options.FDR
        flags.iloc[ambiguous_bcs] = flags.iloc[ambiguous_bcs].where(FDRs > options.FDR, other='cellFinder')
    all_cells['pass'] = passed
    all_cells['flags'] = flags
    return all_cells
    
def filter_cells(all_cells: pd.DataFrame, options: OutlierOptions) -> pd.DataFrame:
    """
    Filter cells based on median absolute deviations of various QC metrics
    'flags' and 'pass' in all_cells are updated in place.

    Args:
        all_cells: Metrics for all cell-barcodes
        options: Options for outlier filtering
        
    Returns:
        MAD statistics used for flagging cells
    """
    cells = all_cells[all_cells['pass']] # Only run outlier filtering on cells (not background)
    flags = pd.Series('', index=cells.index)
    stats = []
    if (options.reads_mads): # Flag cells with low or high total read counts
        lreads = np.log(cells['reads'])
        min_lreads = np.median(lreads) - options.reads_mads * median_abs_deviation(lreads)
        max_lreads = np.median(lreads) + options.reads_mads * median_abs_deviation(lreads)
        flags[lreads < min_lreads] += ';low_reads'
        flags[lreads > max_lreads] += ';high_reads'
        stats.append(('MAD', 'minimum_total_reads', np.round(np.exp(min_lreads))))
        stats.append(('MAD', 'maximum_total_reads', np.round(np.exp(max_lreads))))
    if (options.passing_mads): # Flag cells with low fraction passing reads (counted to a gene)
        preads = cells['passingReads'] / cells['reads']
        min_preads = max(0, np.median(preads) - options.passing_mads * median_abs_deviation(preads))
        flags[preads < min_preads] += ';low_passing_reads'
        stats.append(('MAD', 'minimum_passing_reads', min_preads))
    if (options.mito_mads): # Flag cells with high fraction mito reads
        mito = cells['mitoProp']
        max_mito = max(options.mito_min_thres, np.median(mito) + options.mito_mads * median_abs_deviation(mito))
        flags[mito > max_mito] += ';high_mito'
        stats.append(('MAD', 'maximum_mito', max_mito))
    if options.filter_outliers:
        all_cells.loc[flags.index[flags != ''], 'pass'] = False
    all_cells.loc[cells.index, 'flags'] += flags
    all_cells['flags'] = all_cells['flags'].str.strip(';')
    return pd.DataFrame.from_records(stats, columns=['Category', 'Metric', 'Value'])


def calculate_sample_stats(all_cells: pd.DataFrame, internal_report: bool, is_cellFinder: bool, filter_outliers:bool) -> pd.DataFrame:
    """
    Calculate sample-level statistics from the all_cells dataframe

    Args:
        all_cells: A dataframe containing metrics for all cells in sample
        internal_report: Whether to generate stats consistent with internal report
        is_cellFinder: Was CellFinder used to call cells?
        filter_outliers: Were QC flagged celled filtered / failed?

    Returns:
        Sample-level statistics
    """
    sample_stats = {'Reads': {}, 'Cells': {}, 'Extrapolated Complexity': {}}

    reads = sample_stats['Reads']
    reads['total_reads'] = all_cells['reads'].sum()
    reads['passing_reads'] = all_cells['passingReads'].sum()
    reads['mapped_reads'] = all_cells['mappedReads'].sum()
    reads['mapped_reads_perc'] = reads['mapped_reads'] / reads['total_reads']
    reads['gene_reads'] = all_cells['geneReads'].sum()
    reads['gene_reads_perc'] = reads['gene_reads'] / reads['mapped_reads']
    reads['exon_reads'] = all_cells['exonReads'].sum()
    reads['exon_reads_perc'] = reads['exon_reads'] / reads['mapped_reads']
    reads['antisense_reads'] = all_cells['antisenseReads'].sum()
    reads['antisense_reads_perc'] = reads['antisense_reads'] / reads['mapped_reads']
    reads['mito_reads'] = all_cells['mitoReads'].sum()
    reads['mito_reads_perc'] = reads['mito_reads'] / reads['mapped_reads']
    reads['umis'] = all_cells['umis'].sum()
    reads['saturation'] = 1 - reads['umis'] / reads['passing_reads']
    
    cells = sample_stats['Cells']
    passing_cells = all_cells.loc[all_cells['pass']]
    cells['cells_called'] = passing_cells.shape[0]
    if is_cellFinder:
        cells['cellFinder_calls'] = passing_cells['flags'].str.contains('cellFinder').sum()
    else:
        cells['utc_threshold'] = passing_cells['umis'].min()
    if filter_outliers:
        cells['qc_filtered'] = all_cells['flags'].str.contains('low_').sum() + all_cells['flags'].str.contains('high_').sum()
    else:
        cells['qc_flaged'] = passing_cells['flags'].str.contains('low_').sum() + passing_cells['flags'].str.contains('high_').sum()

    cells['mean_passing_reads'] = passing_cells['reads'].mean()
    cells['median_reads'] = passing_cells['passingReads'].median()
    cells['median_utc'] = passing_cells['umis'].median()
    cells['median_genes'] = passing_cells['genes'].median()
    cells['reads_in_cells'] = passing_cells['passingReads'].sum() / all_cells['passingReads'].sum()
    cells['mito_reads'] = passing_cells['mitoReads'].sum()
    
    complexity = sample_stats['Extrapolated Complexity']
    unique_reads = []
    target_mean_reads = [100, 500, 1000, 5000, 10000, 20000]
    for d in target_mean_reads:
        if d < cells['mean_passing_reads'] or internal_report:
            # Target reads refers to the mean reads per cell, but we are estimating
            # UMIs / complexity in the median cell. Approx. scaling median reads by the same
            # factor as mean reads.
            target_median_reads = (d/cells['mean_passing_reads']) * cells['median_reads']
            unique_reads = extrapolate_unique(cells['median_reads'], cells['median_utc'], target_median_reads)
            complexity[f'target_reads_{d}'] = unique_reads
        else:
            complexity[f'target_reads_{d}'] = np.nan
    
    stats_stats_df = pd.DataFrame([{"Category": category, "Metric": metric, "Value": value} 
                                       for category, metrics in sample_stats.items()
                                       for metric, value in metrics.items()
        ])

    return stats_stats_df


def generate_filtered_matrix(sample_specific_file_paths: dict, all_cells: pd.DataFrame, sample: str) -> None:
    """
    Generates and writes a filtered cell-by-gene expression matrix.
    """
    # Get the correct path structure
    if isinstance(sample_specific_file_paths, dict) and 'GeneFull_Ex50pAS' in sample_specific_file_paths:
        gene_paths = sample_specific_file_paths['GeneFull_Ex50pAS']
    else:
        gene_paths = sample_specific_file_paths  # fallback to original structure

    cells_passing = all_cells['pass'].to_dict()
    filtered_path = f"{sample}_filtered_star_output"
    
    # Create output directory (with parents if needed)
    Path(filtered_path).mkdir(parents=True, exist_ok=True)

    # Verify source files exist
    required_sources = {
        'features': gene_paths['features'],
        'barcodes': gene_paths['barcodes'],
        'mtx': gene_paths['mtx']
    }
    
    for file_key, source_path in required_sources.items():
        if not Path(source_path).exists():
            raise FileNotFoundError(f"Source file {file_key} not found at {source_path}")

    # Copy features file
    features_dest = Path(filtered_path) / "features.tsv"
    shutil.copyfile(required_sources['features'], features_dest)

    # Process matrix and barcodes
    with (open(required_sources['mtx']) as f_raw_mtx,
         open(required_sources['barcodes']) as f_raw_barcodes,
         open("tmp_matrix.mtx", "w") as f_filtered_mtx_tmp,
         open(Path(filtered_path) / "barcodes.tsv", "w") as f_filtered_barcodes):

        raw_barcodes = [line.rstrip() for line in f_raw_barcodes]
        header = f_raw_mtx.readline().strip()
        f_raw_mtx.readline(); f_raw_mtx.readline()  # Skip header lines

        current_barcode = 0
        barcode_row_number = 1
        line_count = 0

        for line in f_raw_mtx:
            split_line = line.split()
            feature, barcode, count = int(split_line[0]), int(split_line[1]), float(split_line[2])
            barcode_sequence = raw_barcodes[barcode - 1]

            if cells_passing[barcode_sequence]:
                if barcode != current_barcode:
                    current_barcode = barcode
                    barcode_row_number += 1
                    f_filtered_barcodes.write(f"{raw_barcodes[current_barcode - 1]}\n")
                
                f_filtered_mtx_tmp.write(f"{feature} {barcode_row_number-1} {count}\n")
                line_count += 1

    # Write final matrix
    header1 = len(pd.read_csv(required_sources['features'], sep="\t", header=None).index)
    header2 = barcode_row_number - 1
    header3 = line_count

    with open(Path(filtered_path) / "matrix.mtx", "w") as f_filtered_mtx:
        f_filtered_mtx.write(f"{header}\n%\n{header1} {header2} {header3}\n")
        if line_count > 0:
            with open("tmp_matrix.mtx") as tmp_file:
                f_filtered_mtx.write(tmp_file.read())

    # Cleanup
    try:
        Path("tmp_matrix.mtx").unlink(missing_ok=True)
    except Exception as e:
        print(f"Warning: Could not remove temp file - {str(e)}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser()
    
    # Required argument for specifying the path to the sampleMetrics CSV file
    parser.add_argument("--sampleMetrics", type=Path, required=True, 
                       help="Path to the sampleMetrics CSV file.") 

    # Modified to support multiple feature types while keeping original defaults
    parser.add_argument("--feature_type", type=str, nargs="+", 
                       required=False, default=['GeneFull_Ex50pAS'],
                       help="STARsolo feature type(s) used.")
    
    # Required and optional arguments for specifying the STARsolo outputs
    parser.add_argument("--STARsolo_out", type=Path, required=True,
                       help="Path to the STARsolo outputs for this sample.")
    parser.add_argument("--matrix_type", type=str, required=False,
                       default='UniqueAndMult-PropUnique.mtx',
                       help="STARsolo matrix type used.")

    # Optional argument to specify the name of the sample
    parser.add_argument("--sample", type=str, required=False,
                       default="example",
                       help="Unique string to identify this sample.")

    # Cell calling parameters (keep all original arguments)
    parser.add_argument("--fixedCells", action="store_true", default=False,
                       help="Fixed number of barcodes to call as cells.")
    parser.add_argument("--expectedCells", type=int, default=0,
                       help="Expected number of cells.")
    parser.add_argument("--topCellPercent", type=int, default=99,
                       help="Cell thresholding parameter.")
    parser.add_argument("--minCellRatio", type=float, default=10,
                       help="Cell thresholding parameter.")
    parser.add_argument("--minUTC", type=int, default=100,
                       help="Minimum unique transcript counts for cell calling.")
    parser.add_argument("--UTC", type=int, default=0,
                       help="Fixed UTC threshold for cell calling.")
    parser.add_argument("--cellFinder", action="store_true",
                       help="Use CellFinder algorithm.")
    parser.add_argument("--FDR", type=float, default=0.001,
                       help="False discovery rate for CellFinder.")
    parser.add_argument("--alpha", type=float, default=None,
                       help="Fixed overdispersion parameter for CellFinder.")
    
    # MAD outlier filtering
    parser.add_argument("--filter_outliers", action="store_true",
                       help="Enable MAD outlier filtering.")
    parser.add_argument("--madsReads", type=float, default=np.nan,
                       help="MAD threshold for total reads.")
    parser.add_argument("--madsPassingReads", type=float, default=np.nan,
                       help="MAD threshold for passing reads fraction.")
    parser.add_argument("--madsMito", type=float, default=np.nan,
                       help="MAD threshold for mito. reads.")
    
    # Optional argument for internal report
    parser.add_argument("--internalReport", action="store_true", default=False)

    args = parser.parse_args()

    # Resolve file paths
    sample_specific_file_paths = io.resolve_sample_specific_file_paths(
        args.STARsolo_out, args.feature_type, args.matrix_type)

    # Initialize options
    call_cells_options = CellCallingOptions(
        fixedCells=args.fixedCells,
        expectedCells=args.expectedCells,
        topCellPercent=args.topCellPercent,
        minCellRatio=args.minCellRatio,
        minUTC=args.minUTC,
        UTC=args.UTC,
        cellFinder=args.cellFinder,
        FDR=args.FDR,
        alpha=args.alpha
    )
    
    outlier_options = OutlierOptions(
        filter_outliers=args.filter_outliers,
        reads_mads=args.madsReads,
        passing_mads=args.madsPassingReads,
        mito_mads=args.madsMito
    )

    # Process gene features
    gene_paths = sample_specific_file_paths['GeneFull_Ex50pAS'] if 'GeneFull_Ex50pAS' in sample_specific_file_paths else sample_specific_file_paths
    all_cells = call_cell_barcodes(gene_paths['mtx'], args.sampleMetrics, options=call_cells_options)
    mad_stats = filter_cells(all_cells, options=outlier_options)
    sample_stats = calculate_sample_stats(
        all_cells, 
        internal_report=args.internalReport,
        is_cellFinder=call_cells_options.cellFinder,
        filter_outliers=outlier_options.filter_outliers
    )
    sample_stats = pd.concat([sample_stats, mad_stats])

    # Write outputs
    metrics_dir = Path(f"{args.sample}_metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    all_cells.to_csv(metrics_dir / f"{args.sample}_allCells.csv")
    sample_stats.to_csv(metrics_dir / f"{args.sample}_sample_stats.csv", index=False)
    
    # Modified call to generate_filtered_matrix
    generate_filtered_matrix(
        sample_specific_file_paths['GeneFull_Ex50pAS'] if 'GeneFull_Ex50pAS' in sample_specific_file_paths else sample_specific_file_paths,
        all_cells,
        args.sample
    )

if __name__ == "__main__":
    main()