# Bangla Image Captioning

<p align="center">
  <img src="data/graph/pytorch_logo.png" width="120" alt="PyTorch Logo">
</p>

A deep learning project for generating Bangla (Bengali) captions from images, built with PyTorch. This repository provides tools for training, evaluating, and using image captioning models on Bangla datasets.

## Features
- End-to-end image captioning in Bangla
- Custom dataset support
- PyTorch-based modular code
- Easy training and evaluation scripts

## Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/Bangla-Image-Captioning.git
   cd Bangla-Image-Captioning
   ```
2. **Install dependencies:**
   - Python 3.6+
   - PyTorch 1.0 or later ([pytorch.org](https://pytorch.org))
   - NLTK
   
   Install with pip:
   ```bash
   pip install torch nltk
   ```
   Then, download NLTK data:
   ```python
   import nltk
   nltk.download()
   ```

## Dataset
- The default dataset is from [BanglaLekha](https://www.banglalekha.org/), but you can use your own dataset.
- **Format:**
  Each line in your CSV should be:
  ```
  /path/to/image1, "caption in Bangla"
  ```
  Example (`train.csv`):
  ```
  0001.jpg, "একজন মেয়ে হাত বাড়িয়ে বৃষ্টির পানি ধরার চেষ্টা করছেন "
  ```

## Usage
- **Training:**
  Edit `config.py` as needed, then run:
  ```bash
  python train.py
  ```
- **Vocabulary Building:**
  ```bash
  python build_vocabulary.py
  ```
- **Custom Dataset:**
  Place your images and CSV in the appropriate folders and update paths in `config.py`.

## Contributing
Contributions, issues, and feature requests are welcome! Feel free to open an issue or submit a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements
- [BanglaLekha](https://www.banglalekha.org/) for the dataset
- PyTorch and NLTK communities
