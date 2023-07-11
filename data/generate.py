import time
import argparse
import pprint as pp
import tqdm

from scipy.spatial.distance import cdist

import numpy as np

from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

def get_optimal_route(distances):
    tsp_size = distances.shape[0]
    num_vehicles = 1
    depot = 0

    # Convert distances to interger (MIP)
    dist_matrix = (10000*distances).astype(int)

    # Create routing model
    if tsp_size > 0:
        manager = pywrapcp.RoutingIndexManager(tsp_size, num_vehicles, depot)
        routing = pywrapcp.RoutingModel(manager)
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        search_parameters.time_limit.FromSeconds(10)

        def distance_callback(from_index, to_index):
            """Returns the distance between the two nodes."""
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return dist_matrix[from_node][to_node]
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)

        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        # Solve the problem.
        assignment = routing.SolveWithParameters(search_parameters)
        if assignment:
            # Only one route here; otherwise iterate from 0 to routing.vehicles() - 1
            route_number = 0

            # Index of the variable for the starting node.
            index = routing.Start(route_number)

            # Variable store
            route = []
            score = .0
            prev = index
            while not routing.IsEnd(index):
                # Convert variable indices to node indices in the displayed route.
                route.append(index)
                score += distances[prev, index]
                index = assignment.Value(routing.NextVar(index))
            return route, score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_nodes", type=int, default=20)
    parser.add_argument("--num_samples", type=int, default=1280)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--filename", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    opts = parser.parse_args()
    
    assert opts.num_samples % opts.batch_size == 0, "Number of samples must be divisible by batch size"
    
    np.random.seed(opts.seed)
    
    if opts.filename is None:
        opts.filename = f"tsp{opts.num_nodes}_ortools.txt"
    
    # Pretty print the run args
    pp.pprint(vars(opts))
    
    with open(opts.filename, "w") as f:
        start_time = time.time()
        for b_idx in tqdm.tqdm(range(opts.num_samples//opts.batch_size)):
            
            idx = 0
            while idx < opts.batch_size:
                nodes_coord = np.random.random([opts.num_nodes, 2])
                nodes_pairwise_dist = cdist(nodes_coord, nodes_coord, metric='euclidean')
                solution, _ = get_optimal_route(nodes_pairwise_dist)
                # Only write instances with valid solutions
                if (np.sort(solution) == np.arange(opts.num_nodes)).all():
                    f.write( " ".join( str(x)+str(" ")+str(y) for x,y in nodes_coord) )
                    f.write( str(" ") + str('output') + str(" ") )
                    f.write( str(" ").join( str(node_idx+1) for node_idx in solution) )
                    f.write( str(" ") + str(solution[0]+1) + str(" ") )
                    f.write( "\n" )
                    idx += 1
            
            assert idx == opts.batch_size
            
        end_time = time.time() - start_time
        
        assert b_idx == opts.num_samples//opts.batch_size - 1
        
    print(f"Completed generation of {opts.num_samples} samples of TSP{opts.num_nodes},.")
    print(f"Total time: {end_time/60:.1f}m")
    print(f"Average time: {end_time/opts.num_samples:.1f}s")