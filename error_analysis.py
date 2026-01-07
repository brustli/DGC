import numpy as np
import torch
import torch.nn.functional as F


def get_emotion_name(label_idx, dataset_name):
    """
    Get emotion name from label index
    """
    if dataset_name == 'IEMOCAP':
        emotion_names = ['Happy', 'Sad', 'Neutral', 'Angry', 'Excited', 'Frustrated']
    elif dataset_name in ['MELD', 'EmoryNLP']:
        emotion_names = ['Neutral', 'Surprise', 'Fear', 'Sadness', 'Joy', 'Disgust', 'Anger']
    elif dataset_name == 'DailyDialog':
        emotion_names = ['Neutral', 'Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']
    else:
        emotion_names = [f'Class_{i}' for i in range(10)]  # Default
    
    if 0 <= label_idx < len(emotion_names):
        return emotion_names[label_idx]
    return f'Unknown_{label_idx}'


def analyze_errors(detailed_info, n_classes, dataset_name, speaker_vocab):
    """
    Analyze classification errors and find highest/lowest confidence errors for each class

    Args:
        detailed_info: Dictionary containing logits, utterances, speakers, lengths, labels, preds
        n_classes: Number of emotion classes
        dataset_name: Name of the dataset
        speaker_vocab: Speaker vocabulary for converting speaker IDs to names

    Returns:
        error_cases: Dictionary containing error analysis results
    """
    if detailed_info is None:
        return None

    # These are lists of batches, not concatenated tensors
    logits_batches = detailed_info['logits']  # List of (B, max_N, C) tensors
    utterances = detailed_info['utterances']  # List of lists
    speakers_batches = detailed_info['speakers']  # List of (B, max_N) tensors
    lengths_batches = detailed_info['lengths']  # List of (B,) tensors
    labels = detailed_info['labels']  # List of lists (with padding -1)
    preds = detailed_info['preds']  # List of lists (with padding)
    
    # Get speaker ID to name mapping
    if hasattr(speaker_vocab, 'itos'):
        speaker_id_to_name = speaker_vocab['itos']
    else:
        speaker_id_to_name = {i: f'Speaker_{i}' for i in range(10)}

    # Store errors for each predicted class
    # errors_by_pred_class[pred_class] = list of error cases
    errors_by_pred_class = {i: [] for i in range(n_classes)}

    # Process each batch
    global_dialog_idx = 0
    for batch_idx in range(len(logits_batches)):
        batch_logits = logits_batches[batch_idx]  # (B, max_N, C)
        batch_speakers = speakers_batches[batch_idx]  # (B, max_N)
        batch_lengths = lengths_batches[batch_idx]  # (B,)

        # Convert logits to probabilities for this batch
        batch_probs = F.softmax(batch_logits, dim=-1)  # (B, max_N, C)

        batch_size = batch_logits.size(0)

        # Iterate through dialogues in this batch
        for local_dialog_idx in range(batch_size):
            dialog_length = batch_lengths[local_dialog_idx].item()
            dialog_utterances = utterances[global_dialog_idx]
            dialog_speakers = batch_speakers[local_dialog_idx]
            dialog_labels = labels[global_dialog_idx]
            dialog_preds = preds[global_dialog_idx]
            dialog_probs = batch_probs[local_dialog_idx]  # (max_N, C)
        
            # Iterate through utterances in this dialogue
            for utt_idx in range(dialog_length):
                true_label = dialog_labels[utt_idx]
                pred_label = dialog_preds[utt_idx]

                # Skip padding
                if true_label == -1:
                    continue

                # Only consider errors (misclassifications)
                if true_label != pred_label:
                    # Get confidence for the predicted (wrong) class
                    pred_confidence = dialog_probs[utt_idx, pred_label].item()

                    # Get all class probabilities
                    all_probs = dialog_probs[utt_idx].cpu().numpy()

                    # Get context (previous and next utterances)
                    context_start = max(0, utt_idx - 3)
                    context_end = min(dialog_length, utt_idx + 4)
                    context_utterances = []

                    for ctx_idx in range(context_start, context_end):
                        speaker_id = dialog_speakers[ctx_idx].item()
                        speaker_name = speaker_id_to_name.get(speaker_id, f'Speaker_{speaker_id}')
                        is_current = (ctx_idx == utt_idx)
                        context_utterances.append({
                            'speaker': speaker_name,
                            'text': dialog_utterances[ctx_idx],
                            'is_current': is_current
                        })

                    # Get current utterance info
                    current_speaker_id = dialog_speakers[utt_idx].item()
                    current_speaker = speaker_id_to_name.get(current_speaker_id, f'Speaker_{current_speaker_id}')
                    current_text = dialog_utterances[utt_idx]

                    # Create error case
                    error_case = {
                        'dialog_idx': global_dialog_idx,
                        'utterance_idx': utt_idx,
                        'speaker': current_speaker,
                        'text': current_text,
                        'context': context_utterances,
                        'true_label': true_label,
                        'true_label_name': get_emotion_name(true_label, dataset_name),
                        'pred_label': pred_label,
                        'pred_label_name': get_emotion_name(pred_label, dataset_name),
                        'pred_confidence': pred_confidence,
                        'all_confidences': all_probs
                    }

                    # Add to the list for this predicted class
                    errors_by_pred_class[pred_label].append(error_case)

            # Move to next dialogue
            global_dialog_idx += 1
    
    # For each predicted class, find highest and lowest confidence errors
    result = {}
    for pred_class in range(n_classes):
        errors = errors_by_pred_class[pred_class]
        
        if len(errors) == 0:
            result[pred_class] = {
                'class_name': get_emotion_name(pred_class, dataset_name),
                'total_errors': 0,
                'highest_confidence': None,
                'lowest_confidence': None
            }
        else:
            # Sort by confidence
            errors_sorted = sorted(errors, key=lambda x: x['pred_confidence'], reverse=True)
            
            result[pred_class] = {
                'class_name': get_emotion_name(pred_class, dataset_name),
                'total_errors': len(errors),
                'highest_confidence': errors_sorted[0],  # Most confident error
                'lowest_confidence': errors_sorted[-1]  # Least confident error
            }
    
    return result


def save_error_analysis(error_results, n_classes, save_path):
    """
    Save error analysis results to a text file
    
    Args:
        error_results: Dictionary from analyze_errors function
        n_classes: Number of emotion classes
        save_path: Path to save the text file
    """
    if error_results is None:
        print("No error analysis results to save.")
        return
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("ERROR ANALYSIS REPORT - Classification Error Cases\n")
        f.write("=" * 100 + "\n\n")
        
        for pred_class in range(n_classes):
            class_info = error_results[pred_class]
            class_name = class_info['class_name']
            total_errors = class_info['total_errors']
            
            f.write("\n" + "=" * 100 + "\n")
            f.write(f"PREDICTED CLASS: {class_name} (Class {pred_class})\n")
            f.write(f"Total Errors: {total_errors}\n")
            f.write("=" * 100 + "\n\n")
            
            if total_errors == 0:
                f.write("No errors for this predicted class.\n\n")
                continue
            
            # Write highest confidence error
            f.write("-" * 100 + "\n")
            f.write("HIGHEST CONFIDENCE ERROR (Most Confident Misclassification)\n")
            f.write("-" * 100 + "\n")
            write_error_case(f, class_info['highest_confidence'], n_classes)
            
            # Write lowest confidence error
            f.write("\n" + "-" * 100 + "\n")
            f.write("LOWEST CONFIDENCE ERROR (Least Confident Misclassification)\n")
            f.write("-" * 100 + "\n")
            write_error_case(f, class_info['lowest_confidence'], n_classes)
            
            f.write("\n")
    
    print(f"Error analysis saved to: {save_path}")


def write_error_case(f, error_case, n_classes):
    """
    Write a single error case to file
    
    Args:
        f: File handle
        error_case: Error case dictionary
        n_classes: Number of emotion classes
    """
    f.write(f"Dialog Index: {error_case['dialog_idx']}\n")
    f.write(f"Utterance Index: {error_case['utterance_idx']}\n")
    f.write(f"Speaker: {error_case['speaker']}\n")
    f.write(f"True Label: {error_case['true_label_name']} (Class {error_case['true_label']})\n")
    f.write(f"Predicted Label: {error_case['pred_label_name']} (Class {error_case['pred_label']})\n")
    f.write(f"Prediction Confidence: {error_case['pred_confidence']:.4f}\n\n")
    
    # Write current utterance
    f.write("Current Utterance:\n")
    f.write(f"  [{error_case['speaker']}]: {error_case['text']}\n\n")
    
    # Write context
    f.write("Dialogue Context (with surrounding utterances):\n")
    for i, ctx in enumerate(error_case['context']):
        if ctx['is_current']:
            f.write(f"  >>> [{ctx['speaker']}]: {ctx['text']} <<<  [CURRENT UTTERANCE]\n")
        else:
            f.write(f"  [{ctx['speaker']}]: {ctx['text']}\n")
    f.write("\n")
    
    # Write all class confidences
    f.write("Confidence Scores for All Classes:\n")
    all_confs = error_case['all_confidences']
    for class_idx in range(len(all_confs)):
        conf = all_confs[class_idx]
        marker = " <-- PREDICTED" if class_idx == error_case['pred_label'] else ""
        marker += " <-- TRUE LABEL" if class_idx == error_case['true_label'] else ""
        f.write(f"  Class {class_idx}: {conf:.4f}{marker}\n")
    f.write("\n")

