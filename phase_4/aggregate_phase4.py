import glob, json, os
import numpy as np
from scipy.stats import ttest_rel

# Similar to previous aggregation but loops over phase_4 structure
# Write code to collect eval_loss/perplexity from phase_4 outputs,
# compute means/stds per dataset/model/schedule, and paired tests.
# Save results to analysis/phase4_stats.json.
