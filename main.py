from testCase import *
from utils import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


def Solution_Exacte():
    choix = int(input("Quelle combinaison choisir ?  a, b = 1, 3  OU  a, b = 1, 1.5 ?  (1/2) \n"))
    while choix != 1 and choix != 2:
        choix = int(input("Quelle combinaison choisir ?  a, b = 1, 3  OU  a, b = 1, 1.5 ?  (1/2)"))

    if choix == 1:
        a, b = 1, 3

    else:
        a, b = 1, 1.5

    Y0 = np.array([0, 1])
    f = brusselator(a, b)
    T = 18
    N = 1000
    t = np.linspace(0, T, N)
    Y = odeint(f, Y0, t)
    plt.xlabel("Temps")
    plt.ylabel("Concentrations")
    plt.grid()
    plt.title("Brusselator - Solution exacte")
    plt.plot(t, Y)
    plt.show()


def Methode_Euler_Explicite():
    choix = int(input("Quelle combinaison choisir ?  a, b = 1, 3  OU  a, b = 1, 1.5 ?  (1/2) \n"))
    while choix != 1 and choix != 2:
        choix = int(input("Quelle combinaison choisir ?  a, b = 1, 3  OU  a, b = 1, 1.5 ?  (1/2)"))

    if choix == 1:
        a, b = 1, 3

    else:
        a, b = 1, 1.5

    f = brusselator(a, b)

    # Conditions initiales et paramètres de la simulation
    Y0 = np.array([0, 1])
    T = 18
    N = 1000
    tau = T / N

    # Boucle pour calculer les approximations Yn à tous les instants tn
    trajectory = [Y0]
    current_Y = Y0

    for n in range(N):
        next_Y = EulerExplicit(current_Y, f, tau)
        trajectory.append(next_Y)
        current_Y = next_Y

    # Conversion de la liste en array numpy pour une manipulation plus facile
    trajectory = np.array(trajectory)

    # Exemple d'utilisation de EulerExplicit avec un vecteur Yn
    Yn = np.array([1, 1])
    tau = 0.01
    Yn_plus_1 = EulerExplicit(Yn, f, tau)
    print(Yn_plus_1)

    # Exemple d'évaluation de la fonction f avec un vecteur Y
    Y = np.array([1, 1])
    result = f(Y)
    print(result)

    # Affichage des premiers éléments de la trajectoire
    print("trajectoire \n", trajectory[:5])

    # Affichage des résultats généraux
    t = np.linspace(0, T, N + 1)
    plt.plot(t, trajectory[:, 0], label="x(t)")
    plt.plot(t, trajectory[:, 1], label="y(t)")
    plt.xlabel("Temps")
    plt.ylabel("Concentrations")
    plt.legend()
    plt.title("Brusselator - Méthode d'Euler explicite ")
    plt.show()

    def concentrationPlotting(t, trajectory):
        plt.plot(t, trajectory[:, 0], label="x(t)")
        plt.plot(t, trajectory[:, 1], label="y(t)")
        plt.xlabel("Temps")
        plt.ylabel("Concentrations")
        plt.legend()
        plt.title("Évolution temporelle des concentrations")
        plt.show()

    def trajectoryPlotting(trajectory):
        plt.plot(trajectory[:, 0], trajectory[:, 1])
        plt.xlabel("x(t)")
        plt.ylabel("y(t)")
        plt.title("Trajectoire des concentrations")
        plt.show()

    # Utilisation des fonctions de visualisation avec les résultats précédents
    t = np.linspace(0, T, N + 1)
    concentrationPlotting(t, trajectory)
    trajectoryPlotting(trajectory)


def Methode_RK4():
    choix = int(input("Quelle combinaison choisir ?  a, b = 1, 3  OU  a, b = 1, 1.5 ?  (1/2) \n"))
    while choix != 1 and choix != 2:
        choix = int(input("Quelle combinaison choisir ?  a, b = 1, 3  OU  a, b = 1, 1.5 ?  (1/2)"))

    if choix == 1:
        a, b = 1, 3

    else:
        a, b = 1, 1.5

    f = brusselator(a, b)

    Y0 = np.array([0, 1])
    T = 18
    N = 1000
    M = 6

    # Méthode RK4 Butcher
    Y = [Y0]
    n = 1
    for t in np.linspace(0, T, N):
        Y.append(RKF4Butcher(f, Y[-1], t / n, M))
        n += 1

    x = []
    y = []
    for i in range(len(Y)):
        x.append(Y[i][0])
        y.append(Y[i][1])

    t = np.linspace(0, T, N + 1)
    plt.plot(t, x, label="x(t)")
    plt.plot(t, y, label="y(t)")
    plt.xlabel("Temps")
    plt.ylabel("Concentrations")
    plt.legend()
    plt.grid()
    plt.title("Brusselator - Méthode RK4")
    plt.show()

    # Méthode RK4 Classique comme en cours
    Ly = ode_RK4(f, 0, T, Y0, N + 1)
    x1 = []
    y1 = []
    for i in range(len(Ly)):
        x1.append(Ly[i][0])
        y1.append(Ly[i][1])

    t = np.linspace(0, T, N + 1)
    plt.plot(t, x1, label="x(t)")
    plt.plot(t, y1, label="y(t)")
    plt.xlabel("Temps")
    plt.ylabel("Concentrations")
    plt.legend()
    plt.grid()
    plt.title("Brusselator - Méthode RK4 classique")
    plt.show()


def Methode_RK4_adapt():
    choix = int(input("Quelle combinaison choisir ?  a, b = 1, 3  OU  a, b = 1, 1.5 ?  (1/2) \n"))
    while choix != 1 and choix != 2:
        choix = int(input("Quelle combinaison choisir ?  a, b = 1, 3  OU  a, b = 1, 1.5 ?  (1/2)"))

    if choix == 1:
        a, b = 1, 3

    else:
        a, b = 1, 1.5

    f = brusselator(a, b)
    Y0 = np.array([0, 1])
    T = 18
    N = 1000
    M = 6

    # Méthode RK45 pas de temps adaptatif
    Y_l = [Y0]
    n = 1
    taux_l = [0]
    for t in np.linspace(0, T, N):
        Y, taux = stepRK45(f, Y_l[-1], t / n, M, epsilon_max=10 ** (-4))
        Y_l.append(Y)
        taux_l.append(taux)
        n += 1

    x2 = []
    y2 = []
    for i in range(len(Y_l)):
        x2.append(Y_l[i][0])
        y2.append(Y_l[i][1])

    t = np.linspace(0, T, N + 1)
    plt.plot(t, x2, label="x(t)")
    plt.plot(t, y2, label="y(t)")
    plt.xlabel("Temps")
    plt.ylabel("Concentrations")
    plt.legend()
    plt.grid()
    plt.title("Brusselator - Méthode RK45 avec pas de temps adaptatif")
    plt.show()

    plt.plot(t, taux_l, label="taux")
    plt.xlabel("Temps")
    plt.ylabel("pas de temps")
    plt.legend()
    plt.grid()
    plt.title("Variation de la valeur des pas de temps en fonction du temps")
    plt.show()


print("-----------------------")
print("  SOLUTION EXACTE")
print("-----------------------")
Solution_Exacte()
print("-----------------------")
print("METHODE EULER EXPLICITE")
print("-----------------------")
Methode_Euler_Explicite()
print("------FIN DE LA METHODE------")
print("-----------------------")
print("METHODE RUNGE-KUTTA ORDRE 4")
print("-----------------------")
Methode_RK4()
print("------FIN DE LA METHODE------")
print("-----------------------")
print("METHODE RUNGE-KUTTA ORDRE 4")
print("AVEC PAS DE TEMPS ADAPTATIF")
print("-----------------------")
Methode_RK4_adapt()
print("------FIN DE LA METHODE------")
