import networkx

def _eps_ball_cardinality(
        R_matrix,
        radius,
        vertex):
    return sum(1 for val in R_matrix[vertex,:] if val <= radius)

def _all_eps_ball_cardinality(
        R_matrix,
        radius):
    return list([_eps_ball_cardinality(R_matrix, radius, v) for v in range(len(R_matrix))])

def min_R_growth(
        R_matrix, 
        radius):
    return min(_all_eps_ball_cardinality(_all_eps_ball_cardinality))

def avg_R_growth(
        R_matrix, 
        radius):
    return avg(_all_eps_ball_cardinality(_all_eps_ball_cardinality))

def max_R_growth(
        R_matrix, 
        radius):
    return max(_all_eps_ball_cardinality(_all_eps_ball_cardinality))

def cdf_R_growth(
        R_matrix,
        radius):
    return (1 / (len(R_matrix) ** 2)) * (
            sum(1 for row in R_matrix for val in row if val <= radius)
        )
