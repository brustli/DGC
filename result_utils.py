import json
import os
import numpy as np
from sklearn.metrics import confusion_matrix


def save_results_json(save_path, results_dict):
    """
    Save experiment results to JSON file
    
    Args:
        save_path: Path to save the JSON file
        results_dict: Dictionary containing all results
    """
    # Convert numpy arrays to lists for JSON serialization
    json_dict = {}
    for key, value in results_dict.items():
        if isinstance(value, np.ndarray):
            json_dict[key] = value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            json_dict[key] = float(value)
        else:
            json_dict[key] = value
    
    # Save to JSON file with pretty formatting
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(json_dict, f, indent=4, ensure_ascii=False)
    
    print(f"Results saved to: {save_path}")


def prepare_test_results(test_new_labels, test_new_preds, test_fscore, test_acc, 
                         fscore_perclass, n_classes, dataset_name, model_path=None):
    """
    Prepare test results dictionary for JSON export
    
    Args:
        test_new_labels: True labels (processed, no padding)
        test_new_preds: Predicted labels (processed, no padding)
        test_fscore: Weighted F1 score
        test_acc: Accuracy
        fscore_perclass: Per-class metrics from classification_report
        n_classes: Number of classes
        dataset_name: Name of the dataset
        model_path: Path to the model (optional, for test mode)
        
    Returns:
        Dictionary containing all results
    """
    # Get emotion names
    if dataset_name == 'IEMOCAP':
        emotion_names = ['Happy', 'Sad', 'Neutral', 'Angry', 'Excited', 'Frustrated']
    elif dataset_name in ['MELD', 'EmoryNLP']:
        emotion_names = ['Neutral', 'Surprise', 'Fear', 'Sadness', 'Joy', 'Disgust', 'Anger']
    elif dataset_name == 'DailyDialog':
        emotion_names = ['Neutral', 'Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']
    else:
        emotion_names = [f'Class_{i}' for i in range(n_classes)]
    
    # Compute confusion matrix
    cm = confusion_matrix(test_new_labels, test_new_preds)

    # Calculate test set class distribution
    total_samples = len(test_new_labels)
    class_distribution = {}

    for i in range(n_classes):
        class_key = str(i)
        # Count samples for this class in test set
        class_count = sum(1 for label in test_new_labels if label == i)
        class_percentage = (class_count / total_samples * 100) if total_samples > 0 else 0

        class_distribution[emotion_names[i]] = {
            'sample_count': class_count,
            'percentage': round(class_percentage, 2)
        }

    # Extract per-class F1 scores
    per_class_f1 = {}
    for i in range(n_classes):
        class_key = str(i)
        if class_key in fscore_perclass:
            per_class_f1[emotion_names[i]] = {
                'f1-score': fscore_perclass[class_key]['f1-score'],
                'precision': fscore_perclass[class_key]['precision'],
                'recall': fscore_perclass[class_key]['recall'],
                'support': fscore_perclass[class_key]['support']
            }
    
    # Prepare results dictionary
    results = {
        'dataset': dataset_name,
        'test_weighted_f1': test_fscore,
        'test_accuracy': test_acc,
        'test_set_class_distribution': class_distribution,
        'per_class_metrics': per_class_f1,
        'confusion_matrix': cm.tolist(),
        'confusion_matrix_labels': emotion_names[:n_classes]
    }
    
    # Add model path if provided (for test mode)
    if model_path:
        results['model_path'] = model_path
    
    # Add macro and weighted averages if available
    if 'macro avg' in fscore_perclass:
        results['macro_avg'] = {
            'f1-score': fscore_perclass['macro avg']['f1-score'],
            'precision': fscore_perclass['macro avg']['precision'],
            'recall': fscore_perclass['macro avg']['recall']
        }
    
    if 'weighted avg' in fscore_perclass:
        results['weighted_avg'] = {
            'f1-score': fscore_perclass['weighted avg']['f1-score'],
            'precision': fscore_perclass['weighted avg']['precision'],
            'recall': fscore_perclass['weighted avg']['recall']
        }
    
    return results


def rename_folder_with_f1(old_path, best_f1):
    """
    Rename folder to include best F1 score
    
    Args:
        old_path: Original folder path
        best_f1: Best F1 score to add to folder name
        
    Returns:
        New folder path
    """
    # Get parent directory and folder name
    parent_dir = os.path.dirname(old_path)
    folder_name = os.path.basename(old_path)
    
    # Create new folder name with F1 score
    new_folder_name = f"{folder_name}_F1_{best_f1:.2f}"
    new_path = os.path.join(parent_dir, new_folder_name)
    
    # Rename folder
    try:
        os.rename(old_path, new_path)
        print(f"Folder renamed: {folder_name} -> {new_folder_name}")
        return new_path
    except Exception as e:
        print(f"Warning: Could not rename folder: {e}")
        return old_path


def create_test_result_folder(base_path, dataset_name):
    """
    Create a timestamped folder for test results
    
    Args:
        base_path: Base path for saved models
        dataset_name: Name of the dataset
        
    Returns:
        Path to the created folder
    """
    from datetime import datetime
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    
    # Create folder path
    folder_name = f"test_{timestamp}"
    folder_path = os.path.join(base_path, dataset_name, folder_name)
    
    # Create folder
    os.makedirs(folder_path, exist_ok=True)
    print(f"Created test result folder: {folder_path}")
    
    return folder_path

