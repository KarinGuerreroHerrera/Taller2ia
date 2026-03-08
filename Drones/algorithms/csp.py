from __future__ import annotations
from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from algorithms.problems_csp import DroneAssignmentCSP


def backtracking_search(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    """
    Basic backtracking search without optimizations.

    Tips:
    - An assignment is a dictionary mapping variables to values (e.g. {X1: Cell(1,2), X2: Cell(3,4)}).
    - Use csp.assign(var, value, assignment) to assign a value to a variable.
    - Use csp.unassign(var, assignment) to unassign a variable.
    - Use csp.is_consistent(var, value, assignment) to check if an assignment is consistent with the constraints.
    - Use csp.is_complete(assignment) to check if the assignment is complete (all variables assigned).
    - Use csp.get_unassigned_variables(assignment) to get a list of unassigned variables.
    - Use csp.domains[var] to get the list of possible values for a variable.
    - Use csp.get_neighbors(var) to get the list of variables that share a constraint with var.
    - Add logs to measure how good your implementation is (e.g. number of assignments, backtracks).

    You can find inspiration in the textbook's pseudocode:
    Artificial Intelligence: A Modern Approach (4th Edition) by Russell and Norvig, Chapter 5: Constraint Satisfaction Problems
    """
    # TODO: Implement your code here
    return recursividadBacktrack(csp, {})
  
def recursividadBacktrack(csp: DroneAssignmentCSP, respuesta: dict[str, str]):
  if csp.is_complete(respuesta):
    return respuesta
  
  variable= csp.get_unassigned_variables(respuesta)[0]
  for valor in csp.domains[variable]:
    if csp.is_consistent(variable, valor, respuesta):
      csp.assign(variable, valor, respuesta)
      resultado= recursividadBacktrack(csp, respuesta)
      
      if resultado is not None:
        return resultado
      csp.unassign(variable, respuesta)
  return None
    


def backtracking_fc(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    """
    Backtracking search with Forward Checking.

    Tips:
    - Forward checking: After assigning a value to a variable, eliminate inconsistent values from
      the domains of unassigned neighbors. If any neighbor's domain becomes empty, backtrack immediately.
    - Save domains before forward checking so you can restore them on backtrack.
    - Use csp.get_neighbors(var) to get variables that share constraints with var.
    - Use csp.is_consistent(neighbor, val, assignment) to check if a value is still consistent.
    - Forward checking reduces the search space by detecting failures earlier than basic backtracking.
    """
    # TODO: Implement your code here
    return recursividadBacktrackFc(csp, {})

def recursividadBacktrackFc(csp: DroneAssignmentCSP, respuesta: dict[str, str]):
  if csp.is_complete(respuesta):
    return respuesta
  
  variable= csp.get_unassigned_variables(respuesta)[0]
  for valor in list(csp.domains[variable]):
    if csp.is_consistent(variable, valor, respuesta):
      csp.assign(variable, valor, respuesta)
      removidos= {}
      fallo= False
      
      for vecino in csp.get_neighbors(variable):
        if vecino in respuesta:
          continue
        removidos[vecino]= []
        
        for val in list(csp.domains[vecino]):
          if not csp.is_consistent(vecino, val, respuesta):
            csp.domains[vecino].remove(val)
            removidos[vecino].append(val)
            
        if len(csp.domains[vecino])==0:
          fallo= True
          break
            
      if not fallo:
        resultado= recursividadBacktrackFc(csp, respuesta)
        if resultado is not None:
          return resultado 
        
      for veci in removidos:
        csp.domains[veci].extend(removidos[veci])    
      csp.unassign(variable, respuesta)
  return None


def backtracking_ac3(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    """
    Backtracking search with AC-3 arc consistency.

    Tips:
    - AC-3 enforces arc consistency: for every pair of constrained variables (Xi, Xj), every value
      in Xi's domain must have at least one supporting value in Xj's domain.
    - Run AC-3 before starting backtracking to reduce domains globally.
    - After each assignment, run AC-3 on arcs involving the assigned variable's neighbors.
    - If AC-3 empties any domain, the current assignment is inconsistent - backtrack.
    - You can create helper functions such as:
      - a values_compatible function to check if two variable-value pairs are consistent with the constraints.
      - a revise function that removes unsupported values from one variable's domain.
      - an ac3 function that manages the queue of arcs to check and calls revise.
      - a backtrack function that integrates AC-3 into the search process.
    """
    # TODO: Implement your code here
    ac3(csp)
    return recursividadBacktrackAc3(csp, {})

def recursividadBacktrackAc3(csp: DroneAssignmentCSP, respuesta:dict[str, str]):
  if csp.is_complete(respuesta):
    return respuesta
  variable= csp.get_unassigned_variables(respuesta)[0]
  
  for valor in list(csp.domains[variable]):
    if csp.is_consistent(variable, valor, respuesta):
      csp.assign(variable, valor, respuesta)
      
      copiaDominios= {}
      for v in csp.variables:
        copiaDominios[v]= list(csp.domains[v])
        
      cola= []
      for n in csp.get_neighbors(variable):
        cola.append((n, variable))
      
      if ac3(csp, deque(cola)):
        resultado= recursividadBacktrackAc3(csp, respuesta)
        if resultado is not None:
          return resultado
        
      csp.domains= copiaDominios
      csp.unassign(variable, respuesta)
  return None

def revisar(csp: DroneAssignmentCSP, xi, xj):
  revisado= False
  for x in list(csp.domains[xi]):
    tieneSoporte= False
    for y in csp.domains[xj]:
      if csp.is_consistent(xi, x, {xj: y}):
        tieneSoporte= True
        break
      
      if not tieneSoporte:
        csp.domains[xi].remove(x)
        revisado= True
  return revisado


def ac3(csp: DroneAssignmentCSP, cola=None):
  if cola is None:
    cola= deque()
    for xi in csp.variables:
      for xj in csp.get_neighbors(xi):
        cola.append((xi, xj))
        
  while cola:
    xi, xj= cola.popleft()
    if revisar(csp, xi, xj):
      if len(csp.domains[xi])==0:
        return False
      
      for xk in csp.get_neighbors(xi):
        if xk!=xj:
          cola.append((xk, xi))
  return True


def backtracking_mrv_lcv(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    """
    Backtracking with Forward Checking + MRV + LCV.

    Tips:
    - Combine the techniques from backtracking_fc, mrv_heuristic, and lcv_heuristic.
    - MRV (Minimum Remaining Values): Select the unassigned variable with the fewest legal values.
      Tie-break by degree: prefer the variable with the most unassigned neighbors.
    - LCV (Least Constraining Value): When ordering values for a variable, prefer
      values that rule out the fewest choices for neighboring variables.
    - Use csp.get_num_conflicts(var, value, assignment) to count how many values would be ruled out for neighbors if var=value is assigned.
    """
    # TODO: Implement your code here (BONUS)
    return recursividadBacktrackMrvLcv(csp, {})
  
def recursividadBacktrackMrvLcv(csp: DroneAssignmentCSP, respuesta: dict[str, str]):
  if csp.is_complete(respuesta):
    return respuesta
  variable= mrv(csp, respuesta)
  
  for valor in lcv(csp, variable, respuesta):
    if csp.is_consistent(variable, valor, respuesta):
      csp.assign(variable, valor, respuesta)
      removidos= {}
      fallo= False
      
      for vecino in csp.get_neighbors(variable):
        if vecino in respuesta:
          continue
        removidos[vecino]= []
        
        for val in list(csp.domains[vecino]):
          if not csp.is_consistent(vecino, val, respuesta):
            csp.domains[vecino].remove(val)
            removidos[vecino].append(val)
            
        if len(csp.domains[vecino])==0:
          fallo= True
          break
            
      if not fallo:
        resultado= recursividadBacktrackFc(csp, respuesta)
        if resultado is not None:
          return resultado 
        
      for veci in removidos:
        csp.domains[veci].extend(removidos[veci])    
      csp.unassign(variable, respuesta)
  return None
  
def mrv(csp: DroneAssignmentCSP, respuesta:dict[str, str]):
  noAsignadas= csp.get_unassigned_variables(respuesta)
  mejorVariable= None
  mejorValor= None
  
  for v in noAsignadas:
    criterio= (len(csp.domains[v]), -len(csp.get_neighbors(v)))
    if mejorValor is None or criterio<mejorValor:
      mejorValor= criterio
      mejorVariable= v
  return mejorVariable

def lcv(csp: DroneAssignmentCSP, variable, respuesta:dict[str, str]):
  valores= []
  for valor in csp.domains[variable]:
    conflictos= csp.get_num_conflicts(variable, valor, respuesta)
    valores.append((conflictos, valor))
    
  valores.sort()
  resultado= []
  for conflictos, valor in valores:
    resultado.append(valor)
  return resultado