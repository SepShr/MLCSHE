import numpy as np
from scipy.spatial.distance import pdist, squareform

# TODO: Add an update_pdist_matrix method to the class.


class PairwiseDistance:
    """Calculates and updates a pairwise distance matrix for a set of complete solutions.
    """

    def __init__(self, vectors: list = [], numeric_ranges: list = [], categorical_indices: list = [], cs_list: list = None) -> None:
        """
        :param vectors: a list of same_length flat vectors.
        :param numeric_ranges: a list of absolute ranges for the numeric parameters.
        :param categorical_indices: a list of indices for catogircal parameters.
        :param cs_list: a list of complete solutions.
        NOTE: Vectors are a list of flattened complete solutions.
        """
        if cs_list is not None:
            self.vectors = np.asarray(self.prepare_for_pdist_eval(cs_list))
        else:
            self.vectors = np.asarray(vectors)
        # normalization factor; NOTE: 1 for a categorical variable
        self.numeric_ranges = np.asarray(numeric_ranges)
        self.categorical_indices = categorical_indices
        self.pdist_matrix = self.calculate_pdist(
            input_array=self.vectors,
            weights=self.numeric_ranges,
            cat_indices=self.categorical_indices
        )

    def calculate_pdist(self, input_array: np.array, weights: np.array, cat_indices: list) -> np.array:
        # Normalize the input data.
        normalized_array = self.normalize_array(input_array, weights)

        # Split nominal and numeric arrays.
        arrays = self.split_array(normalized_array, cat_indices)

        # numeric_dist = pdist(arrays['numeric'], metric='euclidean')  # can be higher than 1 in principle
        # Cannot be higher than 1 in principle.
        numeric_dist = pdist(
            arrays['numeric'], metric='cityblock') / arrays['numeric'].shape[1]
        # Cannot be higher than 1 in principle.
        nominal_dist = pdist(arrays['nominal'], metric='hamming')
        # Distance between Mixed Types (https://www.coursera.org/lecture/cluster-analysis/2-4-distance-betweencategorical-attributes-ordinal-attributes-and-mixed-types-KnvRC)
        return (numeric_dist + nominal_dist) / 2

    @staticmethod
    def split_array(arr: np.array, cat_indexes: list):
        """
        Split an array into two arrays, one for numeric and another one for nominal values.

        :param arr: input array containing both numeric and nominal values
        :param cat_indexes: indexes for categorical (nominal) values
        :return: a dict containing a numeric array and a nominal array
        """

        arrays_dict = {
            'numeric': None,
            'nominal': None
        }

        n_rows, n_cols = arr.shape
        for j in range(n_cols):
            if j in cat_indexes:
                target_array = 'nominal'
            else:
                target_array = 'numeric'

            column = arr[:, j].reshape(n_rows, 1)

            # Incrementally append the target array.
            if arrays_dict[target_array] is not None:
                arrays_dict[target_array] = np.append(
                    arrays_dict[target_array], column, axis=1)
            else:
                arrays_dict[target_array] = column

        return arrays_dict

    @classmethod
    def prepare_for_pdist_eval(cls, cs_list):
        """Trasforms a nested list of complete solutions into a 2D list.
        """
        return [cls.flatten_list(vec) for vec in cs_list]

    @classmethod
    def flatten_list(cls, nested_list):
        """flattens a nested list into a 1D list.
        """
        return sum(map(cls.flatten_list, nested_list), []) if isinstance(nested_list, list) else [nested_list]

    @staticmethod
    def normalize_array(input_array: np.array, weights: np.array) -> np.array:
        return input_array / weights

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return 'Pairwise Distance Matrix = {}'.format(squareform(self.pdist_matrix))
