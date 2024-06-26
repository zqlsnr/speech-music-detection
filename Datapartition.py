import glob
import random
from utils import sort_nicely


def datapartition(mel_dir, val_dir, test_dir, m_train = 2048 * 2):
    """
    Load the individual numpy arrays into partition
    """

    data_trian = glob.glob(mel_dir + "/**/mel-id-[0-9]*.npy", recursive=True)
    sort_nicely(data_trian)

    labels_train = glob.glob(mel_dir+"/**/mel-id-label-[0-9]*.npy", recursive=True)
    sort_nicely(labels_train)

    train_examples = [(data_trian[i], labels_train[i]) for i in range(len(data_trian))]

    random.seed(4)
    random.shuffle(train_examples)

    """
    Creating the train partition.
    """
    # m_train = 20480 * 2
    
    random.seed()
    random.shuffle(train_examples)

    data_MS = glob.glob(mel_dir+ "/**/mel-id-[0-9]*.npy", recursive=True)
    sort_nicely(data_MS)

    labels_MS = glob.glob(mel_dir + "/**/mel-id-label-[0-9]*.npy", recursive=True)
    sort_nicely(labels_MS)

    train_examples_MS = [(data_MS[i], labels_MS[i]) for i in range(len(data_MS))]

    partition = {}
    partition['train'] = train_examples[0:m_train] + train_examples_MS

    random.shuffle(partition['train'])

    print("The size of partition['train'] is {}".format(len(partition['train'])))

    """
    This loads data for the validation set.
    """

    data_val = glob.glob(val_dir + "/**/mel-id-[0-9]*.npy", recursive=True)
    sort_nicely(data_val)

    labels = glob.glob(val_dir + "/**/mel-id-label-[0-9]*.npy", recursive=True)
    sort_nicely(labels)

    validation_examples = [(data_val[i], labels[i]) for i in range(len(data_val))]

    random.seed(4)
    random.shuffle(validation_examples)
    print(validation_examples[0])

    partition['validation'] = validation_examples


    """
    This loads data for the test set.
    """

    data_test = glob.glob(test_dir + "/**/mel-id-[0-9]*.npy", recursive=True)
    sort_nicely(data_test)

    labels_test = glob.glob(test_dir + "/**/mel-id-label-[0-9]*.npy", recursive=True)
    sort_nicely(labels_test)

    test_examples = [(data_test[i], labels_test[i]) for i in range(len(data_test))]

    random.seed(4)
    random.shuffle(test_examples)
    print(test_examples[0])

    partition['test'] = test_examples

    return [partition['train'], partition['validation'], partition['test']]