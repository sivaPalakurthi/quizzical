from csv import DictReader, DictWriter
import argparse

from score_combiner import ScoreCombiner

def main():
    check_accuracy(retrieve_dev_test_entries())

    if args.test_file:
        predictions = make_predictions_on_test_data()

    if args.prediction_file:
        write_predictions(predictions)

def write_predictions(predictions):
    output_file = DictWriter(open(args.prediction_file, 'w'), ['Question ID', 'Answer'])
    output_file.writeheader()

    for entry in sorted(predictions):
        output_file.writerow({
            'Question ID': entry,
            'Answer': predictions[entry][1]})

def make_predictions_on_test_data():
    predictions = {}
    for entry in DictReader(open(args.test_file)):
        predictions[entry['Question ID']] = \
            (entry['Sentence Position'], make_prediction(entry))

    return predictions

def make_prediction(entry):
    return ScoreCombiner(entry['QANTA Scores'], entry['IR_Wiki Scores']).sorted_list()[0][0]

def check_accuracy(entries):
    right = 0
    total = len(entries)
    for entry in entries:
        prediction = make_prediction(entry)
        if prediction == entry['Answer']:
            right += 1

    print("Accuracy on dev: %f" % (float(right) / float(total)))

def retrieve_dev_test_entries():
    entries = []
    for entry in DictReader(open(args.training_file, 'r')):
        if int(entry['Question ID']) % 5 == 0:
            entries.append(entry)

    return entries

def debug(*arguments):
    if args.debug:
        print ' '.join([str(argument) for argument in arguments])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classify quiz bowl training data and predict answers for test data')
    parser.add_argument('training_file', help='input training dataset file')
    parser.add_argument('--test_file', help='input test dataset file - don\'t make predictions if unset')
    parser.add_argument('--prediction_file', help='output prediction file - don\'t write predictions if unset')
    parser.add_argument('--debug', '-d', action='store_true', help='print verbose output for debugging')

    args = parser.parse_args()

    main()

