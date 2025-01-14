from os import makedirs

def create_file(name):
    with open(name, 'w') as f:
        pass

def main():
    # Directories to create
    dirs = [
        'scripts',
        'models',
        'notebooks',
        'data/raw',
        'data/processed',
        'src',
        'results'
    ]

    # Placeholder file to create
    placeholder = '.gitkeep'

    # Create directories
    for d in dirs:
        d = f'../{d}'
        makedirs(d, exist_ok=True)
        create_file(f'{d}/{placeholder}')

main()