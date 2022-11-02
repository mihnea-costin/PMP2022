import pgmpy
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD

# (1) Modelati jocul de carti intr-un model probabilist folosind pgmpy. 

game_model = BayesianNetwork(
    [
        ("Player1", "Player2"),
        ("CarteP2", "DecizieP2"),
        ("CarteP1", "DecizieP1"),
    ]
)

cpd_carte_p1 = TabularCPD(
    variable="CarteP1",
    variable_card=5,
    values=[[0.95, 0.1], [0.05, 0.9], [0.05, 0.1], [0.95, 0.9], [0.05, 0.1]],
)
cpd_carte_p2 = TabularCPD(
    variable="CarteP2",
    variable_card=5,
    values=[[0.95, 0.1], [0.05, 0.9], [0.05, 0.1], [0.95, 0.9], [0.05, 0.1]],
    evidence_card=[1],
)
cpd_decizie_p1 = TabularCPD(
    variable="DecizieP1",
    variable_card=2,
    values=[[0.1, 0.7], [0.9, 0.3]],
    evidence=["CarteP1"],
    evidence_card=[2],
)

cpd_decizie_p2 = TabularCPD(
    variable="DecizieP2",
    variable_card=2,
    values=[[0.1, 0.7], [0.9, 0.3]],
    evidence=["CarteP2","DecizieP1"],
    evidence_card=[2],
)

game_model.add_cpds(
    cpd_carte_p1, cpd_carte_p2, cpd_decizie_p1, cpd_decizie_p2
)

game_model.check_model()

game_model.nodes()

game_model.edges()

game_model.local_independencies("Player1")

game_model.get_independencies()

# (2) Interogati modelul pentru a raspunde la urmatoarele intrebari:
# (c) Schimbati parametrii strategiilor de pariere a celor doi jucatori, astfel incat raspunsul la punctele
# de mai sus sa se schimbe.

game_model = BayesianNetwork(
    [
        ("Player1", "Player2"),
        ("CarteP2", "DecizieP2"),
        ("CarteP1", "DecizieP1"),
    ]
)

cpd_carte_p1 = TabularCPD(
    variable="CarteP1",
    variable_card=5,
    values=[[0.1, 0.9], [0.1, 0.9], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],
)
cpd_carte_p2 = TabularCPD(
    variable="CarteP2",
    variable_card=5,
    values=[[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.1, 0.9]],
    evidence_card=[1],
)
cpd_decizie_p1 = TabularCPD(
    variable="DecizieP1",
    variable_card=2,
    values=[[0.5, 0.5, [0.5, 0.5]],
)

cpd_decizie_p2 = TabularCPD(
    variable="DecizieP2",
    variable_card=2,
    values=[[0.1, 0.9], [0.5, 0.5]],
)

game_model.add_cpds(
    cpd_carte_p1, cpd_carte_p2, cpd_decizie_p1, cpd_decizie_p2
)


# 3) Modificati jocul de mai sus astfel incat ambii jucatori sa primeasca acum cate 2 carti (alternativ), reg-
# ulile de pariere si castig raman aceleasi. Strategiile de pariere a jucatorilor trebuie acum sa ia in
# considerare ambele carti pe care le detine.

game_model = BayesianNetwork(
    [
        ("Player1", "Player2"),
        ("CarteP2", "DecizieP2"),
        ("CarteP1", "DecizieP1"),
    ]
)

cpd_carte_p1 = TabularCPD(
    variable="CarteP1",
    variable_card=5,
    values=[[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],
)
cpd_carte_p2 = TabularCPD(
    variable="CarteP2",
    variable_card=5,
    values=[[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],
    evidence_card=[1],
)
cpd_decizie_p1 = TabularCPD(
    variable="DecizieP1",
    variable_card=2,
    values=[[0.5, 0.5], [0.5, 0.5]],
    evidence=["CarteP1"],
    evidence_card=[2],
)

cpd_decizie_p2 = TabularCPD(
    variable="DecizieP2",
    variable_card=2,
    values=[[0.5, 0.5], [0.5, 0.5]],
    evidence=["CarteP2","DecizieP1"],
    evidence_card=[2],
)

game_model.add_cpds(
    cpd_carte_p1, cpd_carte_p2, cpd_decizie_p1, cpd_decizie_p2
)

game_model.check_model()

game_model.nodes()

game_model.edges()

game_model.local_independencies("Player1")

game_model.get_independencies()
