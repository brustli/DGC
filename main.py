
import numpy as np, argparse, time, random
from datetime import datetime

from model import *

from model_utils import  get_support_set, my_support_set

from plot_utils import plot_confusion_matrix, plot_normalized_confusion_matrix

from trainer import train_or_eval_model

from error_analysis import analyze_errors, save_error_analysis

from result_utils import save_results_json, prepare_test_results, rename_folder_with_f1, create_test_result_folder

from dataloader import get_data_loaders
from transformers import AdamW
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
if hasattr(torch.cuda, 'empty_cache'):
    torch.cuda.empty_cache()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def str2bool(v):
    """ Usage:
    parser.add_argument('--pretrained', type=str2bool, nargs='?', const=True,
                        dest='pretrained', help='Whether to use pretrained models.')
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')



if __name__ == '__main__':
    
    path = './saved_models/'  # 日志 模型保存路径
    parser = argparse.ArgumentParser()

    parser.add_argument('--hidden_dim', type=int, default=300)
    parser.add_argument('--gnn_layers', type=int, default=2, help='Number of gnn layers.')
    parser.add_argument('--diff_layers', type=int, default=4, help='Number of local only layers.')
    parser.add_argument('--local_window', type=int, default=1, help='Size of Local Context information.')
    parser.add_argument('--global_window', type=int, default=5, help='Size of Global Context information.')
    parser.add_argument('--local_dropout', type=float, default=0.1, help='Dropout rate of Local Context information.')
    parser.add_argument('--train_or_test', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--model_path', type=str, default=None)

    parser.add_argument('--CL_type', type=str, default='None', help='Type of CL.')
    parser.add_argument('--temp', type=float, default=0.05, help='temperature parameter.')
    parser.add_argument('--pool_size', type=int, default=256, help='Number of Reps in the Trainset to pick Support set.')
    parser.add_argument('--support_size', type=int, default=64, help='Number of Reps in Pool to generate Support Reps.')
    parser.add_argument('--alpha', type=float, default='0.5', help='Weight of the Contrastive Loss')

    parser.add_argument('--emb_dim', type=int, default=1024, help='Feature size.')
    parser.add_argument('--dataset_name', default='EmoryNLP', type=str, help='dataset name, IEMOCAP, MELD, DailyDialog, EmoryNLP')
    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
    parser.add_argument('--epochs', type=int, default=32, metavar='E', help='number of epochs')

    parser.add_argument('--mlp_layers', type=int, default=2, help='Number of output mlp layers.')

    parser.add_argument('--lr', type=float, default=5e-6, metavar='LR', help='learning rate')  #####
    parser.add_argument('--dropout', type=float, default=0.4, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch_size', type=int, default=16, metavar='BS', help='batch size') ##
    parser.add_argument('--do_CA', type=int, default=1, metavar='CA', help='if Do Cross Attention') ##
    parser.add_argument('--CL_rating', type=float, default=0.5, help='if Do Cross Attention')
    parser.add_argument('--diff_rating', type=float, default=0.4, help='if Do Cross Attention')

    parser.add_argument('--seed', type=int, default=233, help='random seed') ##


    args = parser.parse_args()

    print(args)

    # 固定随机种子X
    seed_everything(args.seed)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", args.device)

    device = args.device
    n_epochs = args.epochs
    batch_size = args.batch_size


    train_loader, valid_loader, test_loader, trainset = get_data_loaders(
        dataset_name=args.dataset_name, batch_size=batch_size, num_workers=0, args=args)


    if 'IEMOCAP' in args.dataset_name:
        n_classes = 6
    else:
        n_classes = 7

    # centers = get_support_set(trainset)
    centers = my_support_set(trainset,n_classes)

    print('building model..')


    if args.train_or_test == 'train':

        current_time = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        print(current_time)  # 示例输出: 2023-10-03 14:30:00
        run_title = 'TRAIN_'+current_time
        path = os.path.join(path, args.dataset_name)
        path = os.path.join(path, run_title)
        os.makedirs(path, exist_ok=True)
    
        model = DualGATs(args, centers, n_classes)

        if torch.cuda.device_count() > 1:
            print('Multi-GPU...........')
            model = nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

        model.to(device)


        loss_function = nn.CrossEntropyLoss(ignore_index=-1) # 忽略掉label=-1 的类


        optimizer = AdamW(model.parameters(), lr=args.lr)

        # best_fscore, best_acc, best_loss, best_label, best_pred, best_mask = None, None, None, None, None, None
        # all_fscore, all_acc, all_loss = [], [], []
        # best_acc = 0.
        # best_fscore = 0.

        # todo 维护一个最佳模型队列（验证和测试分开）：模型按队列存储（index就是epoch数），队列记录所有的f1和loss，队列记录每个epoch输出的聚类绘制特征
        best_test_fscore, best_valid_fscore, best_acc, best_loss, best_label, best_pred, best_mask = None, None, None, None, None, None, None
        all_fscore, all_acc, all_loss = [], [], []
        all_model = []
        all_cluster_plot = []
        all_labels = []
        best_test_fscore = 0.
        best_valid_fscore = 0.
        best_test_epochid = None
        best_valid_epochid = None
        early_stop_count = 0
        best_perclass = None
        best_test_labels = None
        best_test_preds = None
        best_test_detailed_info = None

        best_model = None
        for e in range(n_epochs):  # 遍历每个epoch
            start_time = time.time()

            train_loss, train_acc, _, _, train_fscore, train_emoloss, train_diffloss, train_labelloss, _, _, _, _, _ = train_or_eval_model(model, loss_function,
                                                                            train_loader, device,
                                                                            args, optimizer, True)
            valid_loss, valid_acc, _, _, valid_fscore, valid_emoloss, valid_diffloss, valid_labelloss, _, _, _, _, _ = train_or_eval_model(model, loss_function,
                                                                            valid_loader, device, args)
            test_loss, test_acc, test_label, test_pred, test_fscore, test_emoloss, test_diffloss, test_labelloss, cluster_plot, fscore_perclass, test_new_labels, test_new_preds, test_detailed_info = train_or_eval_model(model, loss_function,
                                                                                          test_loader, device, args)

            # 维护最好成绩和模型队列
            # all_model.append(model.state_dict())
            if test_fscore > best_test_fscore:
                early_stop_count = 0
                best_test_fscore = test_fscore
                best_test_epochid = e

                # Save best results to file with formatted output
                result_log_path = os.path.join(path, "result.txt")
                with open(result_log_path, 'w', encoding='utf-8') as file:
                    file.write("=" * 80 + "\n")
                    file.write("BEST TEST RESULTS\n")
                    file.write("=" * 80 + "\n\n")

                    # Write basic information
                    file.write(f"Dataset: {args.dataset_name}\n")
                    file.write(f"Best Epoch: {e + 1}\n")
                    file.write(f"Best Weighted F1-Score: {best_test_fscore:.2f}%\n")
                    file.write(f"Test Accuracy: {test_acc:.2f}%\n\n")

                    # Write model configuration
                    file.write("-" * 80 + "\n")
                    file.write("Model Configuration:\n")
                    file.write("-" * 80 + "\n")
                    file.write(f"Learning Rate: {args.lr}\n")
                    file.write(f"Dropout: {args.dropout}\n")
                    file.write(f"Batch Size: {args.batch_size}\n")
                    file.write(f"GNN Layers: {args.gnn_layers}\n")
                    file.write(f"Diff Layers: {args.diff_layers}\n")
                    file.write(f"Cross Attention: {bool(args.do_CA)}\n")
                    file.write(f"CL Type: {args.CL_type}\n")
                    file.write(f"Diff Rating: {args.diff_rating}\n")
                    file.write(f"CL Rating: {args.CL_rating}\n\n")

                    # Write per-class F1 scores
                    file.write("-" * 80 + "\n")
                    file.write("Per-Class Performance:\n")
                    file.write("-" * 80 + "\n\n")

                    # Get emotion names based on dataset
                    if args.dataset_name == 'IEMOCAP':
                        emotion_names = ['Happy', 'Sad', 'Neutral', 'Angry', 'Excited', 'Frustrated']
                    elif args.dataset_name in ['MELD', 'EmoryNLP']:
                        emotion_names = ['Neutral', 'Surprise', 'Fear', 'Sadness', 'Joy', 'Disgust', 'Anger']
                    elif args.dataset_name == 'DailyDialog':
                        emotion_names = ['Neutral', 'Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']
                    else:
                        emotion_names = [f'Class_{i}' for i in range(n_classes)]

                    # Write header
                    file.write(f"{'Class':<15} {'Emotion':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n")
                    file.write("-" * 80 + "\n")

                    # Write per-class metrics
                    for i in range(n_classes):
                        class_key = str(i)
                        if class_key in fscore_perclass:
                            metrics = fscore_perclass[class_key]
                            emotion_name = emotion_names[i] if i < len(emotion_names) else f'Class_{i}'
                            precision = metrics.get('precision', 0.0) * 100
                            recall = metrics.get('recall', 0.0) * 100
                            f1 = metrics.get('f1-score', 0.0) * 100
                            support = int(metrics.get('support', 0))

                            file.write(f"{class_key:<15} {emotion_name:<15} {precision:<12.2f} {recall:<12.2f} {f1:<12.2f} {support:<10}\n")

                    file.write("\n" + "-" * 80 + "\n")
                    file.write("Overall Metrics:\n")
                    file.write("-" * 80 + "\n")

                    # Write macro and weighted averages
                    if 'macro avg' in fscore_perclass:
                        macro = fscore_perclass['macro avg']
                        file.write(f"Macro Average:\n")
                        file.write(f"  Precision: {macro.get('precision', 0.0) * 100:.2f}%\n")
                        file.write(f"  Recall:    {macro.get('recall', 0.0) * 100:.2f}%\n")
                        file.write(f"  F1-Score:  {macro.get('f1-score', 0.0) * 100:.2f}%\n\n")

                    if 'weighted avg' in fscore_perclass:
                        weighted = fscore_perclass['weighted avg']
                        file.write(f"Weighted Average:\n")
                        file.write(f"  Precision: {weighted.get('precision', 0.0) * 100:.2f}%\n")
                        file.write(f"  Recall:    {weighted.get('recall', 0.0) * 100:.2f}%\n")
                        file.write(f"  F1-Score:  {weighted.get('f1-score', 0.0) * 100:.2f}%\n\n")

                    if 'accuracy' in fscore_perclass:
                        file.write(f"Accuracy: {fscore_perclass['accuracy'] * 100:.2f}%\n\n")

                    file.write("=" * 80 + "\n")
                    file.write(f"Results saved at: {result_log_path}\n")
                    file.write("=" * 80 + "\n")

                print(f"Best results saved to: {result_log_path}")

                torch.save(model,path + '/best_test_model_CA_' + '_RL_' + str(args.lr) + '_Drop_' + str(args.dropout) +  str(args.do_CA) + '_CL_' + str(args.CL_type) + '_Diff_' + str(args.diff_layers) + '_GL_' + str(args.gnn_layers) + '_DR_' + str(args.diff_rating) + '_CR_' + str(args.CL_rating) +'.pkl')
                best_perclass = fscore_perclass
                best_test_labels = test_new_labels
                best_test_preds = test_new_preds
                best_test_detailed_info = test_detailed_info
            if valid_fscore > best_valid_fscore:
                best_valid_fscore = valid_fscore
                best_valid_epochid = e

            all_fscore.append([valid_fscore, test_fscore])
            all_loss.append([valid_loss, test_loss])
            all_cluster_plot.append(cluster_plot)
            all_labels.append(test_label)

            # print(
            #     'Epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, valid_loss: {}, valid_acc: {}, valid_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'. \
            #     format(e + 1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore, test_loss,
            #            test_acc,
            #            test_fscore, round(time.time() - start_time, 2)))

            print(
                'Epoch: {}, train_loss: {}, train_fscore: {}, valid_loss: {}, valid_fscore: {}, test_loss: {}, test_emoloss: {}, test_diffloss: {}, test_labelloss: {}, test_acc: {}, test_fscore: {}, time: {} sec'. \
                    format(e + 1, train_loss, train_fscore, valid_loss, valid_fscore, test_loss, test_emoloss, test_diffloss, 'None' if args.CL_type == 'None' else test_labelloss,
                           test_acc,
                           test_fscore, round(time.time() - start_time, 2)))

            # torch.save(model.state_dict(), path + args.dataset_name + '/model_' + str(e) + '_' + str(test_acc) + '_' + str(
            #     test_fscore) + '.pkl')
            e += 1

            # early stop
            if test_fscore <= best_test_fscore:
                early_stop_count = early_stop_count+1

            if early_stop_count > 20 and e > 64:
                break


        # 保存测试和验证集的最好性能模型
        # best_model = all_model[best_valid_epochid]
        # torch.save(best_model, path + args.dataset_name + '/best_valid_model_' + str(
        #     e) + '_valid_f1_' + str(
        #     all_fscore[best_valid_epochid][0]) + '_test_f1_' + str(
        #     all_fscore[best_valid_epochid][1])+ '_CA_' + str(args.do_CA) + '_CL_' + str(args.CL_type) + '.pkl')
        # torch.save(best_model,
        #            path + args.dataset_name + '/best_test_model_' + str(e) + '_valid_f1_' + str(
        #                all_fscore[best_test_epochid][0]) + '_test_f1_' + str(
        #                all_fscore[best_test_epochid][1])+ '_CA_' + str(args.do_CA) + '_CL_' + str(args.CL_type) + '_Diff_' + str(args.diff_layers) +'.pkl')

        print(args.dataset_name, ': finish training!')
        print('with args: ', args)


        all_fscore = sorted(all_fscore, key=lambda x: (x[0], x[1]), reverse=True)  # 优先按照验证集 f1 进行排序

        print('Best val F-Score:{}'.format(all_fscore[0][0]))  # 验证集最好性能
        print('Best test F-Score based on validation:{}'.format(all_fscore[0][1]))  # 验证集取得最好性能时 对应测试集的下性能
        print('Best test F-Score based on test:{}'.format(max([f[1] for f in all_fscore])))  # 测试集 最好的性能
        print('Best test epoch on test:{}'.format(best_test_epochid))  # 测试集 最好的性能

        # Plot confusion matrix for best test model
        if best_test_labels is not None and best_test_preds is not None:
            print('\nGenerating confusion matrix for best test model...')
            cm_save_path = path + '/confusion_matrix.png'
            cm_norm_save_path = path + '/confusion_matrix_normalized.png'

            plot_confusion_matrix(best_test_labels, best_test_preds, n_classes,
                                cm_save_path, dataset_name=args.dataset_name)
            plot_normalized_confusion_matrix(best_test_labels, best_test_preds, n_classes,
                                            cm_norm_save_path, dataset_name=args.dataset_name)

        # Perform error analysis for best test model
        if best_test_detailed_info is not None:
            print('\nPerforming error analysis for best test model...')
            error_save_path = path + '/error_analysis.txt'

            # Get speaker vocabulary from test_loader
            speaker_vocab = test_loader.dataset.speaker_vocab

            # Analyze errors
            error_results = analyze_errors(best_test_detailed_info, n_classes,
                                          args.dataset_name, speaker_vocab)

            # Save error analysis
            save_error_analysis(error_results, n_classes, error_save_path)

        # Save results in JSON format
        if best_test_labels is not None and best_test_preds is not None:
            print('\nSaving results in JSON format...')
            result_json_path = path + '/results.json'

            # Prepare results dictionary
            results_dict = prepare_test_results(
                best_test_labels,
                best_test_preds,
                best_test_fscore,
                best_acc if best_acc is not None else test_acc,
                best_perclass,
                n_classes,
                args.dataset_name
            )

            # Save to JSON
            save_results_json(result_json_path, results_dict)

        # Rename folder to include best F1 score
        print('\nRenaming folder to include best F1 score...')
        path = rename_folder_with_f1(path, best_test_fscore)

        if n_classes == 7:
            print('perclass report: '
                  ' class1 : ', best_perclass['0']['f1-score'],
                  ' class2 : ', best_perclass['1']['f1-score'],
                  ' class3 : ', best_perclass['2']['f1-score'],
                  ' class4 : ', best_perclass['3']['f1-score'],
                  ' class5 : ', best_perclass['4']['f1-score'],
                  ' class6 : ', best_perclass['5']['f1-score'],
                  ' class7 : ', best_perclass['6']['f1-score'])
        elif n_classes == 6:
            print('perclass report: '
                  ' class1 : ', best_perclass['0']['f1-score'],
                  ' class2 : ', best_perclass['1']['f1-score'],
                  ' class3 : ', best_perclass['2']['f1-score'],
                  ' class4 : ', best_perclass['3']['f1-score'],
                  ' class5 : ', best_perclass['4']['f1-score'],
                  ' class6 : ', best_perclass['5']['f1-score'])

    else:
        # Test mode: create a new timestamped folder for test results
        print('\n' + '='*80)
        print('TEST MODE')
        print('='*80)

        # Create test result folder with timestamp
        test_result_path = create_test_result_folder('./saved_models', args.dataset_name)
        print(f'DEBUG: test_result_path = {test_result_path}')
        print(f'DEBUG: Folder exists = {os.path.exists(test_result_path)}')

        # Load model
        print(f'\nLoading model from: {args.model_path}')
        model = torch.load(args.model_path)
        model.to(device)

        loss_function = nn.CrossEntropyLoss(ignore_index=-1)  # 忽略掉label=-1 的类

        # Run evaluation
        print('\nRunning evaluation on test set...')
        test_loss, test_acc, test_label, test_pred, test_fscore, test_emoloss, test_diffloss, test_labelloss, cluster_plot, fscore_perclass, test_new_labels, test_new_preds, test_detailed_info = train_or_eval_model(
            model, loss_function, test_loader, device, args)

        print('\n' + '='*80)
        print(f'Test Results:')
        print(f'  Weighted F1-Score: {test_fscore:.2f}%')
        print(f'  Accuracy: {test_acc:.2f}%')
        print('='*80)

        # Save results in JSON format (with model path at the beginning)
        print('\nSaving results in JSON format...')
        result_json_path = os.path.join(test_result_path, 'results.json')
        print(f'DEBUG: result_json_path = {result_json_path}')

        # Prepare results dictionary with model path
        results_dict = prepare_test_results(
            test_new_labels,
            test_new_preds,
            test_fscore,
            test_acc,
            fscore_perclass,
            n_classes,
            args.dataset_name,
            model_path=args.model_path  # Add model path for test mode
        )

        # Save to JSON
        save_results_json(result_json_path, results_dict)
        print(f'DEBUG: JSON file exists = {os.path.exists(result_json_path)}')

        # Plot confusion matrix for test results
        print('\nGenerating confusion matrix...')
        cm_save_path = os.path.join(test_result_path, 'confusion_matrix.png')
        cm_norm_save_path = os.path.join(test_result_path, 'confusion_matrix_normalized.png')
        print(f'DEBUG: cm_save_path = {cm_save_path}')
        print(f'DEBUG: cm_norm_save_path = {cm_norm_save_path}')

        plot_confusion_matrix(test_new_labels, test_new_preds, n_classes,
                            cm_save_path, dataset_name=args.dataset_name)
        plot_normalized_confusion_matrix(test_new_labels, test_new_preds, n_classes,
                                        cm_norm_save_path, dataset_name=args.dataset_name)

        print(f'DEBUG: CM file exists = {os.path.exists(cm_save_path)}')
        print(f'DEBUG: CM norm file exists = {os.path.exists(cm_norm_save_path)}')

        # Perform error analysis for test results
        if test_detailed_info is not None:
            print('\nPerforming error analysis...')
            error_save_path = os.path.join(test_result_path, 'error_analysis.txt')
            print(f'DEBUG: error_save_path = {error_save_path}')

            # Get speaker vocabulary from test_loader
            speaker_vocab = test_loader.dataset.speaker_vocab

            # Analyze errors
            error_results = analyze_errors(test_detailed_info, n_classes,
                                          args.dataset_name, speaker_vocab)

            # Save error analysis
            save_error_analysis(error_results, n_classes, error_save_path)
            print(f'DEBUG: Error analysis file exists = {os.path.exists(error_save_path)}')

        # Print per-class results
        print('\n' + '='*80)
        print('Per-Class Results:')
        print('='*80)
        if n_classes == 7:
            print('perclass report: '
                  ' class1 : ', fscore_perclass['0']['f1-score'],
                  ' class2 : ', fscore_perclass['1']['f1-score'],
                  ' class3 : ', fscore_perclass['2']['f1-score'],
                  ' class4 : ', fscore_perclass['3']['f1-score'],
                  ' class5 : ', fscore_perclass['4']['f1-score'],
                  ' class6 : ', fscore_perclass['5']['f1-score'],
                  ' class7 : ', fscore_perclass['6']['f1-score'])
        elif n_classes == 6:
            print('perclass report: '
                  ' class1 : ', fscore_perclass['0']['f1-score'],
                  ' class2 : ', fscore_perclass['1']['f1-score'],
                  ' class3 : ', fscore_perclass['2']['f1-score'],
                  ' class4 : ', fscore_perclass['3']['f1-score'],
                  ' class5 : ', fscore_perclass['4']['f1-score'],
                  ' class6 : ', fscore_perclass['5']['f1-score'])

        print('\n' + '='*80)
        print(f'All test results saved to: {test_result_path}')
        print('='*80)

        # List all files in the test result folder
        print('\nFiles in test result folder:')
        if os.path.exists(test_result_path):
            files = os.listdir(test_result_path)
            if files:
                for f in files:
                    file_path = os.path.join(test_result_path, f)
                    file_size = os.path.getsize(file_path) if os.path.isfile(file_path) else 0
                    print(f'  - {f} ({file_size} bytes)')
            else:
                print('  (folder is empty)')
        else:
            print('  ERROR: Folder does not exist!')
        print('='*80)

    # # todo 编写绘图作业
    # # matplot绘制loss和f1的折线图
    # loss_f1_plot(all_loss, all_fscore)
    # hpts_plot(all_cluster_plot, best_test_epochid, 'TSNE', name='SPCL_Diff')