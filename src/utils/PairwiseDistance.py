import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from src.utils.utility import flatten_list


class PairwiseDistance:
    """Calculates and updates a pairwise distance matrix for a set of complete solutions.
    """

    def __init__(self, cs_list: list = None, numeric_ranges: list = [], categorical_indices: list = [], weight_num: float = 1.0, weight_cat: float = 1.0) -> None:
        """
        :param numeric_ranges: a list of absolute ranges for the numeric parameters.
        :param categorical_indices: a list of indices for catogircal parameters.
        :param cs_list: a list of complete solutions.
        NOTE: Vectors are a list of flattened complete solutions.
        """
        assert cs_list is not None, "cs_list cannot be None."

        # normalization factor; NOTE: 1 for a categorical variable
        self.numeric_ranges = np.asarray(numeric_ranges)
        self.categorical_indices = categorical_indices
        self.weight_num = weight_num
        self.weight_cat = weight_cat

        if cs_list != []:
            self.vectors = np.asarray(self.prepare_for_dist_eval(cs_list))
            self.cs_list = cs_list
            self.dist_matrix_sq = self.calculate_pdist(
                input_array=self.vectors,
                num_ranges=self.numeric_ranges,
                cat_indices=self.categorical_indices
            )
        else:
            self.vectors = np.asarray([])
            self.dist_matrix_sq = np.asarray([])
            # raise ValueError("cs_list cannot be empty.")
            # self.vectors = np.asarray(vectors)

    def calculate_pdist(self, input_array: np.array, num_ranges: np.array, cat_indices: list) -> np.array:
        # Normalize the input data.
        normalized_array = self.normalize_array(input_array, num_ranges)

        # Split nominal and numeric arrays.
        arrays = self.split_array(normalized_array, cat_indices)

        # numeric_dist = pdist(arrays['numeric'], metric='euclidean')  # can be higher than 1 in principle
        # Cannot be higher than 1 in principle.
        if arrays['numeric'] is None:
            if arrays['nominal'] is None:
                return np.empty(0)
            else:
                nominal_dist = pdist(arrays['nominal'], metric='hamming')
                return squareform(self.weight_cat * nominal_dist)
        else:
            numeric_dist = pdist(
                arrays['numeric'], metric='cityblock') / arrays['numeric'].shape[1]
            if arrays['nominal'] is None:
                return squareform(self.weight_num * numeric_dist)
            else:
                nominal_dist = pdist(arrays['nominal'], metric='hamming')
                return squareform((self.weight_num * numeric_dist + self.weight_cat * nominal_dist) / 2)
        # Cannot be higher than 1 in principle.

        # Distance between Mixed Types (https://www.coursera.org/lecture/cluster-analysis/2-4-distance-betweencategorical-attributes-ordinal-attributes-and-mixed-types-KnvRC)
        # return squareform((self.weight_num * numeric_dist + self.weight_cat * nominal_dist) / 2)

    def calculate_cdist(self, dist_matrix: np.array, calculated_vectors: np.array, new_vectors: np.array, num_ranges: np.array, cat_indices: list):
        # Split to nominal and numeric arrays for both calculated and new vectors.
        calculated_split = self.split_array(
            self.normalize_array(calculated_vectors, num_ranges), cat_indices)
        new_split = self.split_array(
            self.normalize_array(new_vectors, num_ranges), cat_indices)

        if new_split['numeric'] is None:
            if new_split['nominal'] is None:
                raise ValueError(
                    "Both calculated and new vectors cannot be None.")
            else:
                nominal_cdist = cdist(
                    XA=calculated_split['nominal'],
                    XB=new_split['nominal'],
                    metric='hamming')
                final_cdist = (self.weight_cat * nominal_cdist)
        else:
            numeric_cdist = cdist(
                XA=calculated_split['numeric'],
                XB=new_split['numeric'],
                metric='cityblock') / calculated_split['numeric'].shape[1]
            if new_split['nominal'] is None:
                final_cdist = (self.weight_num * numeric_cdist)
            else:
                nominal_cdist = cdist(
                    XA=calculated_split['nominal'],
                    XB=new_split['nominal'],
                    metric='hamming')
                final_cdist = (self.weight_num * numeric_cdist +
                               self.weight_cat * nominal_cdist) / 2

        # numeric_cdist = cdist(
        #     XA=calculated_split['numeric'],
        #     XB=new_split['numeric'],
        #     metric='cityblock') / calculated_split['numeric'].shape[1]
        # nominal_cdist = cdist(
        #     XA=calculated_split['nominal'],
        #     XB=new_split['nominal'],
        #     metric='hamming')
        # final_cdist = (self.weight_num * numeric_cdist +
        #                self.weight_cat * nominal_cdist) / 2

        final_cdist_transpose = np.transpose(final_cdist)

        # Create the top half of the new dist_matrix_sq.
        h_array_1 = np.hstack((dist_matrix, final_cdist))

        # Create the bottom half of the new dist_matrix_sq.
        h_array_2 = np.hstack((final_cdist_transpose, self.calculate_pdist(
            new_vectors, self.numeric_ranges, cat_indices)))

        new_dist_matrix = np.vstack((h_array_1, h_array_2))

        # print(f'calculate_vectors are: {calculated_vectors}')
        # update calculated set of complete solutions
        calculated_vectors = np.append(calculated_vectors, new_vectors, axis=0)

        # return new_dist_matrix, calculated_vectors
        return new_dist_matrix, calculated_vectors

    # def update_dist_matrix(self, new_cs: list, dist_matrix: np.array, calculated_vectors: np.array, num_ranges: np.array, cat_indices: list):
    #     new_vectors = np.asarray(self.prepare_for_dist_eval(new_cs))
    #     self.dist_matrix_sq, self.vectors = self.calculate_cdist(
    #         dist_matrix, calculated_vectors, new_vectors, num_ranges, cat_indices)
    #     self.cs_list += new_cs

    def update_dist_matrix(self, cs_list: list):
        assert cs_list is not None, "The list of complete solutions cannot be None."
        if cs_list != []:
            if self.dist_matrix_sq.size != 0:
                new_vectors = np.asarray(self.prepare_for_dist_eval(cs_list))
                # print(f'new_vectors are: {new_vectors}')
                self.dist_matrix_sq, self.vectors = self.calculate_cdist(
                    self.dist_matrix_sq, self.vectors, new_vectors, self.numeric_ranges, self.categorical_indices)
                self.cs_list += cs_list
            else:
                self.vectors = np.asarray(self.prepare_for_dist_eval(cs_list))
                self.cs_list = cs_list
                self.dist_matrix_sq = self.calculate_pdist(
                    input_array=self.vectors,
                    num_ranges=self.numeric_ranges,
                    cat_indices=self.categorical_indices
                )
        else:
            pass

    def get_distance(self, cs_1, cs_2):
        if cs_1 not in self.cs_list:
            self.update_dist_matrix([cs_1])
        if cs_2 not in self.cs_list:
            self.update_dist_matrix([cs_2])

        return self.dist_matrix_sq[self.cs_list.index(cs_1)][self.cs_list.index(cs_2)]

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
    def prepare_for_dist_eval(cls, cs_list):
        """Trasforms a nested list of complete solutions into a 2D list.
        """
        return [flatten_list(list(vec)) for vec in cs_list]

    @staticmethod
    def normalize_array(input_array: np.array, num_ranges: np.array) -> np.array:
        """Normalize an array of numeric values.
        """
        if len(num_ranges) == 0:
            return input_array
        else:
            return input_array / num_ranges

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return 'Pairwise Distance Matrix = {}'.format(self.dist_matrix_sq)
