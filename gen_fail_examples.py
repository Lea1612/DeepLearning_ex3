import random

MIN_NUMBER = 2
MAX_NUMBER = 999999


def write_pos_examples(pos_examples, variation_name):
    with open('pos_examples_' + variation_name, 'w') as f:
        f.write('\n'.join(pos_examples))


def write_neg_examples(neg_examples, variation_name):
    with open('neg_examples_' + variation_name, 'w') as f:
        f.write('\n'.join(neg_examples))


def generate_positive_prime_numbers(nb_samples):
    positive_examples = []
    while len(positive_examples) < nb_samples:
        i = random.randint(MIN_NUMBER, MAX_NUMBER)

        if i not in positive_examples:
            if is_prime(i):
                positive_examples.append(str(i))

    return positive_examples


def generate_negative_prime_numbers(nb_samples):
    negative_examples = []
    while len(negative_examples) < nb_samples:
        i = random.randint(MIN_NUMBER, MAX_NUMBER)

        if i not in negative_examples:
            if not is_prime(i):
                negative_examples.append(str(i))

    return negative_examples


def is_prime(num):
    for i in range(2, int(num / 2) + 1):
        if (num % i) == 0:
            return False

    return True


def generate_positive_multiple_of_7_numbers(nb_samples):
    positive_examples = []
    while len(positive_examples) < nb_samples:
        i = random.randint(MIN_NUMBER, MAX_NUMBER)

        if i not in positive_examples:
            if i % 7 == 0:
                positive_examples.append(str(i))

    return positive_examples


def generate_negative_multiple_of_7_numbers(nb_samples):
    negative_examples = []
    while len(negative_examples) < nb_samples:
        i = random.randint(MIN_NUMBER, MAX_NUMBER)

        if i not in negative_examples:
            if i % 7 != 0:
                negative_examples.append(str(i))

    return negative_examples


def generate_positive_palindrome(nb_samples, limit):
    positive_samples = []

    for i in range(nb_samples):
        palindrome = ""
        for j in range(limit):
            curr_num = random.randrange(1, 10)
            palindrome += str(curr_num)
        positive_samples.append(palindrome + palindrome[::-1])
    return positive_samples


def generate_negative_palindrome(nb_samples, limit):
    negative_examples = []

    for i in range(nb_samples):
        palindrome = ""
        for j in range(limit):
            curr_num = random.randrange(1, 10)
            palindrome += str(curr_num)
        negative_examples.append(palindrome + palindrome)
    return negative_examples


if __name__ == '__main__':
    number_of_samples = 500

    pos_prime_examples = generate_positive_prime_numbers(number_of_samples)
    neg_prime_examples = generate_negative_prime_numbers(number_of_samples)
    write_pos_examples(pos_prime_examples, "prime")
    write_neg_examples(neg_prime_examples, "prime")

    pos_multiple_of_7_examples = generate_positive_multiple_of_7_numbers(number_of_samples)
    neg_multiple_of_7_examples = generate_negative_multiple_of_7_numbers(number_of_samples)
    write_pos_examples(pos_multiple_of_7_examples, "multiple_7")
    write_neg_examples(neg_multiple_of_7_examples, "multiple_7")

    pos_palindrome_examples = generate_positive_palindrome(number_of_samples, 20)
    neg_palindrome_examples = generate_negative_palindrome(number_of_samples, 20)
    write_pos_examples(pos_palindrome_examples, "palindrome")
    write_neg_examples(neg_palindrome_examples, "palindrome")
