[Advances in Financial Machine Learning](https://www.amazon.com/Advances-Financial-Machine-Learning-Marcos/dp/1119482089) by Marcos López de Prado
is a must read for anyone interested in quantitative trading.

All credit for the algorithms, concepts, and original implementation goes to him.

This repository hosts a small selection of code from this book,
along with faster implementations (in Cython).

**Note**: This code does not come with any warranty.
It may contain bugs and you are responsible for reviewing and testing it before using it in any critical applications.

### Setup
Running `python setup.py build_ext --inplace` should compile the cython code into importable modules.

### Requirements
1. Cython
2. Pandas
3. Numpy
