from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.artifact_entity  import ClassificationMetricArtifact
import sys

def get_classification_score(y_true, y_pred) -> ClassificationMetricArtifact:
    """
    Method Name :   get_classification_score
    Description :   This function calculates classification metrics such as accuracy, f1 score, precision, and recall.
    
    Output      :   Returns a ClassificationMetricArtifact object containing the calculated metrics.
    On Failure  :   Write an exception log and then raise an exception
    """
    try:
        logging.info("Calculating classification metrics.")
        
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)

        logging.info(f"Metrics calculated: Accuracy={accuracy}, F1 Score={f1}, Precision={precision}, Recall={recall}")

        return ClassificationMetricArtifact(
            f1_score=f1,
            precision_score=precision,
            recall_score=recall
        )
    except Exception as e:
        raise NetworkSecurityException(e,sys)