import json
import random
import pandas as pd
random.seed(12345)


def data_split(all_data_address, fold_num=10):
    with open(all_data_address, 'r') as f:
        all_json_data = json.load(f)
    other_list, why_list, use_list, done_list, property_list, what_list = [], [], [], [], [], []
    for idx, code, comment, label in zip(all_json_data['id'], all_json_data['code'], all_json_data['comment'],
                                         all_json_data['label']):
        if label == 'noisy':
            continue
        if label == 'others':
            other_list.append((idx, code, comment, label))
        elif label == 'why':
            why_list.append((idx, code, comment, label))
        elif label.strip() == 'how-to-use':
            use_list.append((idx, code, comment, label))
        elif label == 'how-it-is-done':
            done_list.append((idx, code, comment, label))
        elif label == 'property':
            property_list.append((idx, code, comment, label))
        else:
            what_list.append((idx, code, comment, label))

    print(len(other_list), len(why_list), len(use_list), len(done_list), len(property_list), len(what_list))
    random.shuffle(other_list)
    random.shuffle(why_list)
    random.shuffle(use_list)
    random.shuffle(done_list)
    random.shuffle(property_list)
    random.shuffle(what_list)

    other_num = len(other_list) // fold_num
    why_num = len(why_list) // fold_num
    use_num = len(use_list) // fold_num
    done_num = len(done_list) // fold_num
    property_num = len(property_list) // fold_num
    what_num = len(what_list) // fold_num

    other_folds = []
    why_folds = []
    use_folds = []
    done_folds = []
    property_folds = []
    what_folds = []
    for i in range(fold_num):
        if i == fold_num - 1:
            other_folds.append(other_list[i * other_num:])
            why_folds.append(why_list[i * why_num:])
            use_folds.append(use_list[i * use_num:])
            done_folds.append(done_list[i * done_num:])
            property_folds.append(property_list[i * property_num:])
            what_folds.append(what_list[i * what_num:])
        else:
            other_folds.append(other_list[i * other_num:(i + 1) * other_num])
            why_folds.append(why_list[i * why_num:(i + 1) * why_num])
            use_folds.append(use_list[i * use_num:(i + 1) * use_num])
            done_folds.append(done_list[i * done_num:(i + 1) * done_num])
            property_folds.append(property_list[i * property_num:(i + 1) * property_num])
            what_folds.append(what_list[i * what_num:(i + 1) * what_num])

    train_folds = []
    test_folds = []
    for i in range(fold_num):
        train = []
        test = []
        for j in range(fold_num):
            if j == i:
                test.extend(other_folds[j])
                test.extend(why_folds[j])
                test.extend(use_folds[j])
                test.extend(done_folds[j])
                test.extend(property_folds[j])
                test.extend(what_folds[j])
            else:
                train.extend(other_folds[j])
                train.extend(why_folds[j])
                train.extend(use_folds[j])
                train.extend(done_folds[j])
                train.extend(property_folds[j])
                train.extend(what_folds[j])
        train_folds.append(train)
        test_folds.append(test)

    return train_folds, test_folds


def make_data(file_address):
    fold_num = 10
    train_folds, test_folds = data_split(file_address, fold_num)
    for i in range(fold_num):
        train = train_folds[i]
        test = test_folds[i]
        random.shuffle(train)
        train_dict = {'id': [t[0] for t in train], 'code': [t[1] for t in train],
                      'comment': [t[2] for t in train], 'label': [t[3] for t in train]}
        print(len(train_dict['id']))

        random.shuffle(test)
        test_dict = {'id': [t[0] for t in test], 'code': [t[1] for t in test],
                     'comment': [t[2] for t in test], 'label': [t[3] for t in test]}
        print(len(test_dict['id']))

        with open(f"./10fold/train_{i}.json", "w") as f:
            json.dump(train_dict, f)
        with open(f"./10fold/test_{i}.json", "w") as f:
            json.dump(test_dict, f)


def remove_noisy(all_data_address):
    df = pd.read_excel(all_data_address)
    output_dict = {'id': [], 'code': [], 'comment': [], 'label': []}
    for idx, code, comment, label in zip(list(df['id']), list(df['code']), list(df['comment']), list(df['label'])):
        if label == 'noisy':
            continue
        else:
            output_dict['id'].append(idx)
            output_dict['code'].append(code)
            output_dict['comment'].append(comment)
            output_dict['label'].append(label)

    print(len(output_dict['id']), len(output_dict['code']), len(output_dict['comment']), len(output_dict['label']))

    with open('./comment_data_final.json', 'w') as w:
        json.dump(output_dict, w)


if __name__ == "__main__":
    remove_noisy('./manually_labeled_data_20000.xlsx')
    make_data('./comment_data_final.json')
