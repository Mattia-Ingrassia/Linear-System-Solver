import matplotlib.pyplot as plt
from Solver import Solver
import os

SAVE_PATH = "results"

def create_plot(data):
    errors = []
    methods = []
    
    for result in data:
        for solution in result["solutions"]:
            #print(result.solutions)
            methods.append(solution["solver_name"])
            errors.append(solution["relative_error"])
    
    plt.figure(figsize=(12,8))
    plt.title("Errore relativo per tolleranza 10e-4")
    plt.xlabel("Metodi")
    plt.ylabel("errore")
    plt.plot(methods, errors)
    plt.grid(True)
    plt.show()

def create_confront_plot(data, matrix_name):
  
    if not os.path.exists(f"{SAVE_PATH}/{matrix_name}/images"):       
        os.makedirs(f"{SAVE_PATH}/{matrix_name}/images")
    
    path = f"{SAVE_PATH}/{matrix_name}/images"

    results_iterations = {}
    results_time = {}
    for method in Solver.Method:
        # if method != Solver.Method.GRADIENT:
            results_iterations[method.value] = []
            results_time[method.value] = []

    tolerances = []
    for element in data:
        tolerances.append(element["tolerance"])
        for solution in element["solutions"]:
            # if solution["solver_name"] != Solver.Method.GRADIENT.value:

                results_iterations[solution["solver_name"]].append(solution["iterations"])
                results_time[solution["solver_name"]].append(solution["time_spent"])

    markers = ['o', 's', '^', 'd']

    #print(results_iterations)

    plt.figure(figsize=(10, 6))
    
    marker_index = 0
    for method in results_iterations.keys():
        plt.plot(tolerances, results_iterations[method] , marker=markers[marker_index], label=method)

        marker_index += 1

    # Opzionalmente inverti l’asse x se vuoi vedere le tolleranze in ordine decrescente
    plt.gca().invert_xaxis()

    # Aggiungi etichette e legenda
    plt.xlabel('Tolleranza (log scale)')
    plt.ylabel('Numero di iterazioni')
    plt.title('Confronto metodi per diverse tolleranze')
    plt.xscale('log')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{path}/{matrix_name}_tol_nit_confronto.png", dpi = 300)

    plt.show()

    plt.figure(figsize=(10, 6))
    
    marker_index = 0
    for method in results_time.keys():
        plt.plot(tolerances, results_time[method] , marker=markers[marker_index], label=method)

        marker_index += 1

    # Opzionalmente inverti l’asse x se vuoi vedere le tolleranze in ordine decrescente
    plt.gca().invert_xaxis()



    # Aggiungi etichette e legenda
    plt.xlabel('Tolleranza (log scale)')
    plt.ylabel('Tempo impiegato')
    plt.title('Confronto metodi per diverse tolleranze')
    plt.xscale('log')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{path}/{matrix_name}_tol_tempo_confronto.png", dpi = 300)

    plt.show()


    print(results_time)
    print("-----")
    print(results_iterations)

def create_sparsity_plot(data, matrix_name):
    if not os.path.exists(f"{SAVE_PATH}/{matrix_name}"):       
        os.makedirs(f"{SAVE_PATH}/{matrix_name}")
    
    path = f"{SAVE_PATH}/{matrix_name}"
    plt.figure(figsize=(10, 10))
    plt.title(f"Distribuzione elementi di {matrix_name}")
    plt.spy(data, markersize=0.15)
    plt.savefig(f"{path}/sparsity_pattern.png", dpi=300)
    plt.show()