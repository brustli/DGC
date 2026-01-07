import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
plt.switch_backend('agg')


def plot_confusion_matrix(y_true, y_pred, n_classes, save_path, dataset_name='IEMOCAP'):
    """
    Plot and save confusion matrix heatmap

    Args:
        y_true: True labels
        y_pred: Predicted labels
        n_classes: Number of emotion classes
        save_path: Path to save the confusion matrix figure
        dataset_name: Name of the dataset for title
    """
    # Define emotion labels for different datasets
    if dataset_name == 'IEMOCAP':
        class_names = ['Happy', 'Sad', 'Neutral', 'Angry', 'Excited', 'Frustrated']
    elif dataset_name in ['MELD', 'EmoryNLP']:
        class_names = ['Neutral', 'Surprise', 'Fear', 'Sadness', 'Joy', 'Disgust', 'Anger']
    elif dataset_name == 'DailyDialog':
        class_names = ['Neutral', 'Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']
    else:
        # Default: use class indices
        class_names = [f'Class {i}' for i in range(n_classes)]

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Create figure with larger size for better readability
    plt.figure(figsize=(12, 10))

    # Plot heatmap with improved formatting
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': 'Count'},
                linewidths=0.5,
                linecolor='gray',
                square=True,
                annot_kws={'size': 11})

    plt.title(f'Confusion Matrix - {dataset_name}', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()

    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Confusion matrix saved to: {save_path}')
    plt.close()

    return cm


def plot_normalized_confusion_matrix(y_true, y_pred, n_classes, save_path, dataset_name='IEMOCAP'):
    """
    Plot and save normalized confusion matrix heatmap (percentage)

    Args:
        y_true: True labels
        y_pred: Predicted labels
        n_classes: Number of emotion classes
        save_path: Path to save the confusion matrix figure
        dataset_name: Name of the dataset for title
    """
    # Define emotion labels for different datasets
    if dataset_name == 'IEMOCAP':
        class_names = ['Happy', 'Sad', 'Neutral', 'Angry', 'Excited', 'Frustrated']
    elif dataset_name in ['MELD', 'EmoryNLP']:
        class_names = ['Neutral', 'Surprise', 'Fear', 'Sadness', 'Joy', 'Disgust', 'Anger']
    elif dataset_name == 'DailyDialog':
        class_names = ['Neutral', 'Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']
    else:
        # Default: use class indices
        class_names = [f'Class {i}' for i in range(n_classes)]

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Normalize by row (true labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Convert to percentage (0-100 scale)
    cm_percentage = cm_normalized * 100

    # Create figure with larger size for better readability
    plt.figure(figsize=(12, 10))

    # Plot heatmap with percentage values and improved formatting
    sns.heatmap(cm_percentage, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': 'Percentage (%)'},
                vmin=0, vmax=100,
                linewidths=0.5,
                linecolor='gray',
                square=True,
                annot_kws={'size': 11})

    plt.title(f'Normalized Confusion Matrix - {dataset_name}', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()

    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Normalized confusion matrix saved to: {save_path}')
    plt.close()

    return cm_normalized


# import sklearn
# import numpy as np
# import  matplotlib.pyplot as plt
# import hypertools as hyp
# plt.switch_backend('agg')
#
#
# # hypertools的降维可视化
# def hpts_plot(feature, epochid, method = 'TSNE', name = 'pca'):
#
#     feature = feature[epochid]
#     feature_for_plot = np.array(feature[0][:400])
#     label_for_plot = np.array(feature[1][:400])
#     hyp.plot(feature_for_plot, '.', reduce=method, ndims=2, hue=label_for_plot, save_path='./'+name+'.png')
#     # pl t.title('TSNE')
#
#
# def loss_f1_plot(losses, fscores):
#
#     # todo 两张图（loss和f1），每张图两条曲线（valid和test），用实例
#
#     plt.title("losses")
#     plt.xlabel("epochs")
#     plt.ylabel("loss value")
#     y1_l = np.array([v[0] for i, v in enumerate(losses)])
#     x1_l = np.array([i for i, v in enumerate(losses)])
#     y2_l = np.array([v[1] for i, v in enumerate(losses)])
#     x2_l = np.array([i for i, v in enumerate(losses)])
#
#     f, (loss_p, f1_p) = plt.subplots(1, 2, sharey=True)
#     loss_p.plot(x1_l, y1_l, "b+", label="valid losses")
#     loss_p.plot(x2_l, y2_l, "r+", label="test losses")
#     loss_p.legend()
#     loss_p.set_title('losses')
#     loss_p.set_xlabel("epochs")
#     loss_p.set_ylabel("loss value")
#
#     y1_f = np.array([v[0] for i, v in enumerate(fscores)])
#     x1_f = np.array([i for i, v in enumerate(fscores)])
#     y2_f = np.array([v[1] for i, v in enumerate(fscores)])
#     x2_f = np.array([i for i, v in enumerate(fscores)])
#
#     f1_p.plot(x1_f, y1_f, "b+", label="valid fscores")
#     f1_p.plot(x2_f, y2_f, "r+", label="test fscores")
#     f1_p.legend()
#     f1_p.set_title('fscores')
#     f1_p.set_xlabel("epochs")
#     f1_p.set_ylabel("f1 value")
#
#     plt.show()
#     plt.savefig('loss&f1.png', c='c')
#
#
# def draw_fig(list, name, epoch):
#     # 我这里迭代了200次，所以x的取值范围为(0，200)，然后再将每次相对应的准确率以及损失率附在x上
#     x1 = range(1, epoch + 1)
#     print(x1)
#     y1 = list
#     if name == "loss":
#         plt.cla()
#         plt.title('Train loss vs. epoch', fontsize=20)
#         plt.plot(x1, y1, '.-')
#         plt.xlabel('epoch', fontsize=20)
#         plt.ylabel('Train loss', fontsize=20)
#         plt.grid()
#         plt.savefig("./lossAndacc/Train_loss.png")
#         plt.show()
#     elif name == "acc":
#         plt.cla()
#         plt.title('Train accuracy vs. epoch', fontsize=20)
#         plt.plot(x1, y1, '.-')
#         plt.xlabel('epoch', fontsize=20)
#         plt.ylabel('Train accuracy', fontsize=20)
#         plt.grid()
#         plt.savefig("./lossAndacc/Train _accuracy.png")
#         plt.show()

