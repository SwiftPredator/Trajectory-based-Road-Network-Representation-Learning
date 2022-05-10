# import json

# import overpass


# class OverpassWrapper:
#     def __init__(self, timeout: int = 600, debug: bool = True):
#         self.api = overpass.API(timeout=timeout, debug=debug)

#     def execute_query(
#         self, query: str, save_path: str = None, verbosity: str = "body"
#     ) -> dict:
#         data = self.api.get(query, verbosity=verbosity, responseformat="json")

#         if save_path:
#             with open(save_path, "w") as fp:
#                 json.dump(data, fp)

#         return data
