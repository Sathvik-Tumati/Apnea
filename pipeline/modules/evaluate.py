import json
import logging
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, classification_report
from CLI.db.database import insert_apnea_results
from .config import NumpyEncoder
logger = logging.getLogger(__name__)

