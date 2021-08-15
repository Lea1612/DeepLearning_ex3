try:
    from xeger import Xeger
except ImportError:
    import subprocess
    import os
    import sys

    cfolder = os.path.dirname(os.path.abspath(__file__))
    cmd = [sys.executable, "-m", "pip", "install", "--target=" + cfolder, 'xeger']
    subprocess.call(cmd)
    from xeger import Xeger


def generate_positive_samples(regex, nb_samples, limit):
    positive_samples = []
    x = Xeger(limit=limit)
    for i in range(nb_samples):
        positive_samples.append(x.xeger(regex))
    return positive_samples


def generate_negative_samples(regex, nb_samples, limit):
    negative_samples = []
    x = Xeger(limit=limit)
    for i in range(nb_samples):
        negative_samples.append(x.xeger(regex))
    return negative_samples


def write_pos_examples(pos_examples):
    with open('pos_examples', 'w') as f:
        f.write('\n'.join(pos_examples))


def write_neg_examples(neg_examples):
    with open('neg_examples', 'w') as f:
        f.write('\n'.join(neg_examples))


if __name__ == '__main__':
    number_of_samples = 500
    limit = 15
    positiveRegex = "[1-9]+a+[1-9]+b+[1-9]+c+[1-9]+d+[1-9]+"
    negativeRegex = "[1-9]+a+[1-9]+c+[1-9]+b+[1-9]+d+[1-9]+"
    pos_examples = generate_positive_samples(positiveRegex, number_of_samples, limit)
    neg_examples = generate_negative_samples(negativeRegex, number_of_samples, limit)
    write_pos_examples(pos_examples)
    write_neg_examples(neg_examples)
