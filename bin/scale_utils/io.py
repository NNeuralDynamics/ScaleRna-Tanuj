"""Utilities related to file I/O and parsing"""

import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Union, List


def readJSON(file: Path, preserveDictOrder: bool = False):
    """[Keep exactly the same as original]"""
    with open(file) as f:
        str = f.read()
        strStripped = str.rstrip()
        pairs_hook = OrderedDict if preserveDictOrder else None
        parsedJSON = json.loads(strStripped, object_pairs_hook=pairs_hook)
    return parsedJSON

def ensurePathsExist(filePaths: Dict[str, Path]):
    """[Keep exactly the same as original]"""
    for key, value in filePaths.items():
        if not value.exists():
            raise FileNotFoundError(f"{key} was assumed to be located at '{str(value)}'. It is missing")

def resolve_sample_specific_file_paths(STARsolo_out: Path, 
                                    feature_type: Union[str, List[str]], 
                                    matrix_type: str) -> Dict[str, Dict[str, Path]]:
    """
    Returns paths to necessary files in STARsolo output directory.
    
    Modified to handle both single feature type (original) or multiple feature types.
    """
    # Convert single feature_type to list for uniform handling
    feature_types = [feature_type] if isinstance(feature_type, str) else feature_type
    
    file_paths = {}
    for ft in feature_types:
        mtx_prefix = STARsolo_out / ft
        
        # Common files for all feature types
        files = {
            'features': mtx_prefix / "raw" / "features.tsv",
            'barcodes': mtx_prefix / "raw" / "barcodes.tsv",
            'summary': mtx_prefix / "Summary.csv"
        }
        
        # Special handling for matrix and stats files
        if ft == "SJ":
            files['stats'] = mtx_prefix / "Features.stats"
            # SJ doesn't use matrix file
            if 'mtx' in files:  
                del files['mtx']
        else:
            files['mtx'] = mtx_prefix / "raw" / matrix_type
            files['stats'] = mtx_prefix / "CellReads.stats"
        
        # Only check paths for files we actually need
        required_files = {k:v for k,v in files.items() if not (ft == "SJ" and k == "mtx")}
        ensurePathsExist(required_files)
        
        file_paths[ft] = files
    
    # Maintain backward compatibility - if single feature_type was passed, return its files directly
    if isinstance(feature_type, str):
        return file_paths[feature_type]
    return file_paths
