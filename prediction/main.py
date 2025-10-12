import argparse
import predict

def main():

    parser = argparse.ArgumentParser(description='Process Poscar file and charge state.')
    parser.add_argument('poscar_path', help='Path to the Poscar file')
    parser.add_argument('charge_state', type=int, help='Charge state')
    parser.add_argument('--output', default='predictions.txt', help='Output file for predictions')

    args = parser.parse_args()

    poscar_file = args.poscar_path
    charge = args.charge_state
    output_file = args.output

    predictions = predict.predict(poscar_file, charge)

    with open(output_file, 'w') as f:
        for label, value in predictions.items():
            f.write(f"{label}: {value}\n")

if __name__ == "__main__":
    main()
