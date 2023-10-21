import dateutil.parser
import re
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score, top_k_accuracy_score

from constants import CURRENT_TIME_KEYWORDS, BACHELORS, MASTERS, MBA, PHD, ASSOCIATE, HIGH_SCHOOL, CERTIFICATION, BACHELORS_SHORT, MASTERS_SHORT, CURRENT_DATE, LABELS


def process_time_string(time_string):
    '''
    Convert a time string to a datetime object if possible; otherwise, return None.
    
    Args:
        time_string (str): The input time string.

    Returns:
        datetime.datetime or None: The parsed datetime object or None if parsing fails.
    '''
    try:
        time_string = time_string.strip()
        if time_string.lower() in CURRENT_TIME_KEYWORDS:
            return CURRENT_DATE
        else:
            date = dateutil.parser.parse(time_string)
            return date
    except:
        return None


def extract_time(time_range):
    '''
    Extract start_time and end_time from a time_range.

    Args:
        time_range (list): A list containing one or two time strings.

    Returns:
        tuple: A tuple containing (start_time, end_time).
    '''    
    start_time, end_time = None, None
    if isinstance(time_range, (list, tuple)):
        if len(time_range) == 1:
            end_time = process_time_string(time_range[0])
        elif len(time_range) == 2:
            start_time, end_time = (process_time_string(time_range[0]), process_time_string(time_range[1]))
            if start_time is not None and end_time is None:
                end_time = CURRENT_DATE
        return start_time, end_time
    else:
        return start_time, end_time

def check_degree_type(degree):
    '''
    Check the type of degree based on the input text.

    Args:
        degree (str): The degree text.

    Returns:
        str: The identified degree type ('MBA', 'PhD', 'Masters', 'Bachelors', 'Associate', 'High School', 'Certification', 'Other').
    '''
    degree = clean_text(degree)
    if any(mba in degree for mba in MBA):
        return 'MBA'
    elif any(phd in degree for phd in PHD):
        return 'PhD'
    elif any(masters in degree for masters in MASTERS) or any(masters_short in degree[:6] for masters_short in MASTERS_SHORT):
        return 'Masters'
    elif any(bachelors in degree for bachelors in BACHELORS) or any (bachelors_short in degree[:6] for bachelors_short in BACHELORS_SHORT):
        return 'Bachelors'
    elif any(associate in degree for associate in ASSOCIATE):
        return 'Associate'
    elif any(high_school in degree for high_school in HIGH_SCHOOL):
        return 'High School'
    elif any(certification in degree for certification in CERTIFICATION):
        return 'Certification'
    else:
        return 'Other'


def clean_text(text):
    '''
    Clean and preprocess the input text.

    Args:
        text (str): The input text.

    Returns:
        str: The cleaned and preprocessed text.
    '''
    try:
        text = str(text)
        # Replace everything but alphanumeric characters, periods, and spaces with a space
        text = re.sub(r'[^A-Za-z0-9 .]+', ' ', text)
        # Remove extra spaces
        text = ' '.join(text.split())
        # Remove periods
        text = text.replace('.', '')
        # Convert to lower case
        text = text.lower()
        return text
    except:
        return ''


def compute_metrics(labels, predictions, prob_predictions,  model_name, split='val'):
    '''
    Compute and print various classification metrics, and generate confusion matrix plots.

    Args:
        labels (array-like): The true labels.
        predictions (array-like): The predicted labels.
        split (str): The name of the dataset split (e.g., 'train', 'val', 'test').
    '''
    # Accuracy
    accuracy = accuracy_score(labels, predictions)
    print(f'{split} accuracy: {accuracy}')

    # Accuracy at top 2
    top2_accuracy = top_k_accuracy_score(labels, prob_predictions, k=2, labels=np.arange(len(LABELS)))
    print(f'{split} top 2 accuracy: {top2_accuracy}')

    # Accuracy at top 3
    top3_accuracy = top_k_accuracy_score(labels, prob_predictions, k=3, labels=np.arange(len(LABELS)))
    print(f'{split} top 3 accuracy: {top3_accuracy}')

    # Precision
    precision = precision_score(labels, predictions, average=None)
    for label, score in zip(LABELS, precision):
        print(f'{split} Precision for {label}: {score}')

    # Recall
    recall = recall_score(labels, predictions, average=None)
    print(f'{split} Recall: {recall}')
    for label, score in zip(LABELS, recall):
        print(f'{split} Recall for {label}: {score}')

    # F1 score
    f1 = f1_score(labels, predictions, average=None)
    print(f'{split} F1: {f1}')
    for label, score in zip(LABELS, f1):
        print(f'{split} F1 for {label}: {score}')

    # Macro F1 score
    macro_f1 = f1_score(labels, predictions, average='macro')
    print(f'{split} Macro F1: {macro_f1}')

    # Weighted F1 score
    weighted_f1 = f1_score(labels, predictions, average='weighted')
    print(f'{split} Weighted F1: {weighted_f1}')

    # Confusion matrix
    confusion = confusion_matrix(labels, predictions)
    print(f'{split} Confusion matrix: {confusion}')

    # Normalized confusion matrix
    normalized_confusion = confusion.astype('float') / confusion.sum(axis=0)[:, np.newaxis]
    normalized_confusion = np.round(normalized_confusion, 2)
    print(f'{split} Normalized confusion matrix: {normalized_confusion}')

    for matrix, title in zip([confusion, normalized_confusion], ['Confusion matrix', 'Normalized confusion matrix']):
        # Plot confusion matrix with labels
        plt.figure(figsize=(10, 10))
        plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Reds)
        plt.title(title)
        plt.colorbar()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        tick_marks = np.arange(len(LABELS))
        plt.xticks(tick_marks, LABELS, rotation=45)
        plt.yticks(tick_marks, LABELS)
        plt.tight_layout()
        plt.savefig(f'{model_name}_{split}_{title}.png')
        plt.show()
