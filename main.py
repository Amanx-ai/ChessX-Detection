import argparse
import sys
import os

# Add src to the system path to allow easy imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    parser = argparse.ArgumentParser(description="ChessX Detection: Run the AI or Data Tools.")
    parser.add_argument('command', choices=['gui', 'train', 'label'], 
                        help="Command to run: 'gui' (launch application), 'train' (start model training), or 'label' (start data labeling tool).")
    
    args = parser.parse_args()

    # Use os.system with the full path to ensure executability regardless of environment
    if args.command == 'gui':
        gui_path = os.path.join(os.path.dirname(__file__), 'src', 'gui_app.py')
        os.system(f"{sys.executable} {gui_path}")

    elif args.command == 'train':
        train_path = os.path.join(os.path.dirname(__file__), 'src', 'training', 'train.py')
        os.system(f"{sys.executable} {train_path}")
        
    elif args.command == 'label':
        label_path = os.path.join(os.path.dirname(__file__), 'src', 'data', 'label_tool.py')
        os.system(f"{sys.executable} {label_path}")

if __name__ == '__main__':
    main()
