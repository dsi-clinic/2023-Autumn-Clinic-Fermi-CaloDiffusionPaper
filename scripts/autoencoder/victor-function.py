import math

def calculate_euclidean_distance(v1, v2):
    sum_var = 0
    for val1,val2 in zip(v1,v2):
        sum_var += ((val1-val2)**2)
    return sum_var**.5

def find_k_nearest_neighbor(point, dataset, k):
    return dataset[k]

def dimension_estimation(all_data, k, n):
    left = 1/(n*(k-1))
    value_sum = 0
    for i in range(1,n+1):
        for j in range(1, k-1+1):
            nearest_k_x_i = find_k_nearest_neighbor(all_data[i], all_data, k)
            nearest_j_x_i = find_k_nearest_neighbor(all_data[i], all_data, j)
            numerator = calculate_euclidean_distance(all_data[i], nearest_k_x_i)
            denominator = calculate_euclidean_distance(all_data[i], nearest_j_x_i)
            value_sum += (math.log(numerator/denominator))
    return 1/(left*value_sum)



