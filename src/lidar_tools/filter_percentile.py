import numpy as np
import scipy


# This method accepts a percentile threshold as part of pdal args, converts it to a Z-score
# and then filters out points that fall outside this range. We use a nodata value (-9999) for
# points that fall outside of the specified threshold
def filter_percentile(ins, outs):
    # pdal args is defined in the PDAL pipeline that calls this script
    percentile_threshold = pdalargs["percentile_threshold"]  # noqa: F821
    z = ins["Z"]
    
    # Check if array is empty or has insufficient data
    if z.size == 0:
        # Return all points as-is if no data
        outs["Classification"] = ins["Classification"]
        return True
    
    # Remove NaN values for computation
    z_valid = z[~np.isnan(z)]
    
    if z_valid.size == 0:
        # All values are NaN, return original classifications
        outs["Classification"] = ins["Classification"]
        return True
    
    z_val = scipy.stats.norm.ppf(percentile_threshold)
    mean = np.nanmean(z)
    std = np.nanstd(z)
    z_scores = (z - mean) / std
    filtered_classification = np.where(z_scores > z_val, 18, ins["Classification"])
    outs["Classification"] = filtered_classification
    return True