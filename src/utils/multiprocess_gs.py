
# from tuning.grid_search import train_and_save, 

# def gs_worker(args):
#     """
#     Worker for one Monte Carlo replication.
#     """
#     return train_and_save(params=args)


# def multiprocessing_mc():
#     """
#     Main function to run Monte Carlo simulations using multiprocessing.
#     """
  
#     args =

    
#     # Use multiprocessing to run MC replications concurrently.
#     with Pool(processes=4) as pool:
#         results = pool.map(mc_worker, rep_args)
#         # Collect results
#         for res in results:
#             print(f"Seed: {res['seed']} | CV Error: {res['CV_error']}")
#     # Optionally, save results to a file.

