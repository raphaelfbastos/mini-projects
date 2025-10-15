from typing import Hashable

class DFA:

    def __init__(self, states:set, alphabet:set, transition_function:dict, start_state:Hashable, final_states:set):
        """
        Creates a Deterministic Finite Automaton M=(Q, Σ, δ, s, F), where\n
        Q = set of states\n
        Σ = set of input symbols\n
        δ = transition function, δ: Q × Σ → Q\n
        s = start state, s ∈ Q\n
        F = set of final states, F ⊆ Q
        """
        self.states = states
        self.alphabet = alphabet
        self.transition_function = transition_function
        self.start_state = start_state
        self.final_states = final_states

    def extended_transition_function(self, q:Hashable, w:str):
        """
        δ^: Q × Σ* → Q
        """
        if w:
            return self.transition_function[self.extended_transition_function(q, w[:-1]), w[-1]]
        return q
    
    def accepts(self, w:str):
        """
        DFA M accepts w if δ^(s, w) ∈ F
        """
        return self.extended_transition_function(self.start_state, w) in self.final_states
    
    def minimize(self):
        """
        Minimizes |Q|
        """
        reachable = [self.start_state]
        for q in reachable:
            for a in self.alphabet:
                qa = self.transition_function[q, a]
                if qa not in reachable:
                    reachable.append(qa)
        distinguishable = {}
        for i in range(len(reachable)-1):
            for j in range(i+1, len(reachable)):
                p, q = reachable[i], reachable[j]
                distinguishable[frozenset({p, q})] = (p in self.final_states) ^ (q in self.final_states)
        new = True
        while new:
            new = False
            for pair in distinguishable:
                if not distinguishable[pair]:
                    p, q = pair
                    for a in self.alphabet:
                        pa = self.transition_function[p, a]
                        qa = self.transition_function[q, a]
                        if pa != qa and distinguishable[frozenset({pa, qa})]:
                            distinguishable[pair] = True
                            new = True
                            break
        equivalent = {q: {q} for q in reachable}
        for pair in distinguishable:
            if not distinguishable[pair]:
                p, q = pair
                equivalent[p].add(q)
                equivalent[q].add(p)
        return DFA(set(map(frozenset, equivalent.values())), self.alphabet, {(frozenset(equivalent[q]), a): frozenset(equivalent[self.transition_function[q, a]]) for a in self.alphabet for q in reachable}, frozenset(equivalent[self.start_state]), {frozenset(equivalent[q]) for q in self.final_states if q in reachable})
    
    def __mul__(self, other:DFA):
        return ProductMachine(self, other)
    
    def __add__(self, other:DFA):
        """
        Constructs a Product Machine M=(Q, Σ, δ, s, F) from DFAs M1=(Q1, Σ, δ1, s1, F1) and M2=(Q2, Σ, δ2, s2, F2), where\n
        Q = Q1 × Q2\n
        δ: Q × Σ → Q\n
        s = <s1, s2>\n
        F = {<p, q> | p ∈ F1 ∨ q ∈ F2}
        """
        product_machine = ProductMachine(self, other)
        product_machine.final_states = {(p, q) for (p, q) in product_machine.states if p in product_machine.M1.final_states or q in product_machine.M2.final_states}
        return product_machine
    
    def __neg__(self):
        """
        Constructs a DFA M=(Q, Σ, δ, s, F) from DFA M'=(Q, Σ, δ, s, F'), where\n
        F = Q - F'
        """
        return DFA(self.states, self.alphabet, self.transition_function, self.start_state, self.states.difference(self.final_states))

class ProductMachine(DFA):

    def __init__(self, M1:DFA, M2:DFA):
        """
        Constructs a Product Machine M=(Q, Σ, δ, s, F) from DFAs M1=(Q1, Σ, δ1, s1, F1) and M2=(Q2, Σ, δ2, s2, F2), where\n
        Q = Q1 × Q2\n
        δ: Q × Σ → Q\n
        s = <s1, s2>\n
        F = F1 × F2
        """
        self.M1 = M1
        self.M2 = M2
        start = (M1.start_state, M2.start_state)
        pairs = [start]
        transitions = {}
        for pair in pairs:
            p, q = pair
            for a in M1.alphabet:
                pa = self.M1.transition_function[p, a]
                qa = self.M2.transition_function[q, a]
                new_pair = (pa, qa)
                transitions[pair, a] = new_pair
                if new_pair not in pairs:
                    pairs.append(new_pair)
        super().__init__(pairs, M1.alphabet, transitions, start, {(p, q) for p in self.M1.final_states for q in self.M2.final_states})
    
    def extended_transition_function(self, r:tuple, w:str):
        """
        δ^(<p, q>, w) = <δ^1(p, w), δ^2(q, w)>
        """
        p, q = r
        return self.M1.extended_transition_function(p, w), self.M2.extended_transition_function(q, w)

class NFA:

    def __init__(self, states:set, alphabet:set, transition_function:dict, start_states:set, final_states:set):
        """
        Creates a Nondeterministic Finite Automaton N=(Q, Σ, Δ, S, F), where\n
        Q = set of states\n
        Σ = set of input symbols\n
        Δ = transition function, Δ: Q × Σ → 𝒫(Q)\n
        S = set of start states, S ⊆ Q\n
        F = set of final states, F ⊆ Q
        """
        self.states = states
        self.alphabet = alphabet
        self.transition_function = transition_function
        self.start_states = start_states
        self.final_states = final_states

    def extended_transition_function(self, A:set, w:str):
        """
        Δ^: 𝒫(Q) × Σ* → 𝒫(Q)
        """
        if w:
            return {q for p in self.extended_transition_function(A, w[:-1]) if (p, w[-1]) in self.transition_function for q in self.transition_function[p, w[-1]]}
        return A
    
    def accepts(self, w:str):
        """
        NFA N accepts w if Δ^(S, w) ∩ F ≠ ∅
        """
        return bool(self.extended_transition_function(self.start_states, w).intersection(self.final_states))
    
    def subset_construction(self):
        """
        Constructs a DFA M=(Q, Σ, δ, s, F) from NFA N=(Q', Σ, Δ, S', F'), where\n
        Q = 𝒫(Q')\n
        δ(A, w) = Δ^(A, w)\n
        s = S'\n
        F = {A ⊆ Q' | A ∩ F' ≠ ∅}
        """
        start = frozenset(self.start_states)
        subsets = [start]
        transitions = {}
        final = set({start}) if start.intersection(self.final_states) else set()
        for subset in subsets:
            for a in self.alphabet:
                new_subset = frozenset(self.extended_transition_function(subset, a))
                transitions[subset, a] = new_subset
                if new_subset not in subsets:
                    subsets.append(new_subset)
                    if new_subset.intersection(self.final_states):
                        final.add(new_subset)
        return DFA(set(subsets), self.alphabet, transitions, start, final)
    
class eNFA:

    def __init__(self, states:set, alphabet:set, transition_function:dict, start_states:set, final_states:set):
        """
        Creates a Nondeterministic Finite Automaton N=(Q, Σ, Δ, S, F) with ε-transitions, where\n
        Q = set of states\n
        Σ = set of input symbols\n
        Δ = transition function, Δ: Q × Σ → 𝒫(Q)\n
        S = set of start states, S ⊆ Q\n
        F = set of final states, F ⊆ Q
        """
        self.states = states
        self.alphabet = alphabet
        self.transition_function = transition_function
        self.start_states = start_states
        self.final_states = final_states

    def epsilon_closure(self, A:set):
        """
        ε-Closure(A) = ∪ ε-Closure(q)
        """
        states = list(A)
        for p in states:
            if (p, '') in self.transition_function:
                for q in self.transition_function[p, '']:
                    if q not in states:
                        states.append(q)
        return set(states)

    def extended_transition_function(self, A:set, w:str):
        """
        Δ^: 𝒫(Q) × Σ* → 𝒫(Q) = {A ⊆ Q | A = ε-Closure(A)}
        """
        if w:
            return {q for p in self.extended_transition_function(A, w[:-1]) if (p, w[-1]) in self.transition_function for q in self.epsilon_closure(self.transition_function[p, w[-1]])}
        return self.epsilon_closure(A)
    
    def accepts(self, w:str):
        """
        ε-NFA N accepts w if Δ^(S, w) ∩ F ≠ ∅
        """
        return bool(self.extended_transition_function(self.start_states, w).intersection(self.epsilon_closure(self.final_states)))
    
    def subset_construction(self):
        """
        Constructs a DFA M=(Q, Σ, δ, s, F) from ε-NFA N=(Q', Σ, Δ, S', F'), where\n
        Q = 𝒫(Q') = {A ⊆ Q | A = ε-Closure(A)}\n
        δ(A, w) = ε-Closure(Δ^(A, w))\n
        s = ε-Closure(S')\n
        F = {A ⊆ Q' | A ∩ ε-Closure(F') ≠ ∅}
        """
        start = frozenset(self.epsilon_closure(self.start_states))
        subsets = [start]
        transitions = {}
        final = set({start}) if start.intersection(self.final_states) else set()
        epsilon_final = self.epsilon_closure(self.final_states)
        for subset in subsets:
            for a in self.alphabet:
                new_subset = frozenset(self.extended_transition_function(subset, a))
                transitions[subset, a] = new_subset
                if new_subset not in subsets:
                    subsets.append(new_subset)
                    if new_subset.intersection(epsilon_final):
                        final.add(new_subset)
        return DFA(set(subsets), self.alphabet, transitions, start, final)

def RegExpr2eNFA(regular_expression:str):
    """
    Constructs an ε-NFA N from Regular Expression r such that L(N) = L(r).\n
    Given Regular Expressions r1 and r2, the following are also Regular Expressions:\n
    r1.r2\n
    r1+r2\n
    r1*\n
    (r1)
    """
    evaluation_stack = []
    operator_stack = []
    operators = {'(': 3, '*': 2, '.': 1, '+': 0}
    alphabet = set()
    for r in regular_expression:
        if r not in operators and r != ')':
            evaluation_stack.append(r)
            alphabet.add(r)
        else:
            if r == ')':
                s = operator_stack.pop()
                while s != '(':
                    evaluation_stack.append(s)
                    s = operator_stack.pop()
            else:
                while operator_stack and operator_stack[-1] != '(' and operators[r] <= operators[operator_stack[-1]]:
                    evaluation_stack.append(operator_stack.pop())
                operator_stack.append(r)
    while operator_stack:
        evaluation_stack.append(operator_stack.pop())
    eNFAs = []
    i = 0
    for r in evaluation_stack:
        match r:
            case '*':
                states1, transitions1, start1, final1 = eNFAs.pop()
                eNFAs.append(({i} | states1 | {i+1}, {(i, ''): start1 | {i+1}} | transitions1 | {(q, ''): {i+1} for q in final1} | {(i+1, ''): {i}}, {i}, {i+1}))
                i += 2
            case '.':
                states2, transitions2, start2, final2 = eNFAs.pop()
                states1, transitions1, start1, final1 = eNFAs.pop()
                eNFAs.append((states1 | states2, transitions1 | {(q, ''): start2 | transitions1.get((q, ''), set()) for q in final1} | transitions2, start1, final2))
            case '+':
                states2, transitions2, start2, final2 = eNFAs.pop()
                states1, transitions1, start1, final1 = eNFAs.pop()
                eNFAs.append(({i} | states1 | states2, {(i, ''): start1 | start2} | transitions1 | transitions2, {i}, final1 | final2))
                i += 1
            case _:
                eNFAs.append(({i, i+1}, {(i, r): {i+1}}, {i}, {i+1}))
                i += 2
    states, transition_function, start_states, final_states = eNFAs[0]
    return eNFA(states, alphabet, transition_function, start_states, final_states)

def eNFA2RegExpr(N:eNFA): # work in progress
    """
    work in progress
    """
    renamed = dict(zip(N.states, range(len(N.states))))
    states = set(range(len(N.states)))
    transition_function = {}
    for (p, a) in N.transition_function:
        transition_function[renamed[p], a] = {renamed[q] for q in N.transition_function[p, a]}
    transition_function['s', ''] = {renamed[q] for q in N.start_states}
    for q in {renamed[q] for q in N.final_states}:
        transition_function[q, ''] = {'f'}
    for p in states:
        for r in {transition_function[q, a] for (q, a) in transition_function if q == p}:
            pass # get all possible states from p 

def printDFA(M:DFA):
    M = renameDFA(M)
    print(f'Q = {{{', '.join(map(str, range(len(M.states))))}}}')
    print(f'Σ = {{{', '.join(sorted(list(M.alphabet)))}}}')
    for (p, a), q in sorted(M.transition_function.items(), key=lambda x:x[0]):
        print(f'δ({p}, {a}) = {q}')
    print(f's = {M.start_state}')
    print(f'F = {{{', '.join(map(str, M.final_states))}}}')

def renameDFA(M:DFA):
    visited = [M.start_state]
    alphabet = sorted(list(M.alphabet))
    for q in visited:
        for a in alphabet:
            qa = M.transition_function[q, a]
            if qa not in visited:
                visited.append(qa)
    renamed = dict(zip(visited, range(len(visited))))
    transitions = {}
    for (q, a) in M.transition_function:
        transitions[renamed[q], a] = renamed[M.transition_function[q, a]]
    return DFA(renamed.values(), M.alphabet, transitions, renamed[M.start_state], {renamed[q] for q in M.final_states})

def printNFA(N:NFA): # doesn't print in visit order
    renamed = dict(zip(N.states, range(len(N.states))))
    transition_function = {}
    for (p, a) in N.transition_function:
        transition_function[renamed[p], a] = {renamed[q] for q in N.transition_function[p, a]}
    print(f'Q = {{{', '.join(map(str, range(len(N.states))))}}}')
    print(f'Σ = {{{', '.join(N.alphabet)}}}')
    for (q, a) in transition_function:
        print(f'Δ({q}, {a if a else 'ε'}) = {{{', '.join(map(str, transition_function[q, a]))}}}')
    print('S = {' + ', '.join(map(str, {renamed[q] for q in N.start_states})) + '}')
    print('F = {' + ', '.join(map(str, {renamed[q] for q in N.final_states})) + '}')

# M1 = DFA({'A', 'B', 'C'}, {'0', '1'}, {('A', '0'): 'B', ('A', '1'): 'A', ('B', '0'): 'C', ('B', '1'): 'B', ('C', '0'): 'C', ('C', '1'): 'C'}, 'A', {'C'})
# M2 = DFA({'S', 'A', 'B', 'C', 'D'}, {'0'}, {('S', '0'): 'A', ('A', '0'): 'B', ('B', '0'): 'C', ('C', '0'): 'D', ('D', '0'): 'A'}, 'S', {'B', 'D'})
# M3 = DFA({'0', '1', '2', '3', '4', '5', '6', '7'}, {'a', 'b'}, {('0', 'a'): '1', ('0', 'b'): '2', ('1', 'a'): '3', ('1', 'b'): '4', ('2', 'a'): '4', ('2', 'b'): '3', ('3', 'a'): '5', ('3', 'b'): '5', ('4', 'a'): '5', ('4', 'b'): '5', ('5', 'a'): '5', ('5', 'b'): '5', ('6', 'a'): '3', ('6', 'b'): '5', ('7', 'a'): '5', ('7', 'b'): '4'}, '0', {'1', '2', '5'})
# Q1 = M3.minimize()

# N1 = NFA({'A', 'B'}, {'a', 'b'}, {('A', 'a'): {'A'}, ('A', 'b'): {'B'}}, {'A'}, {'B'}) # has any number of a's followed by a single b
# C1 = N1.subset_construction()

# M00 = DFA({'A', 'B', 'C'}, {'0', '1'}, {('A', '0'): 'B', ('A', '1'): 'A', ('B', '0'): 'C', ('B', '1'): 'A', ('C', '0'): 'C', ('C', '1'): 'C'}, 'A', {'C'}) # contains 00
# M02 = DFA({'A', 'B'}, {'0', '1'}, {('A', '0'): 'B', ('A', '1'): 'A', ('B', '0'): 'A', ('B', '1'): 'B'}, 'A', {'A'}) # has an even number of 0's
# P1 = -M00 * M02 # does not contain 00 AND has an even number of 0's
# P2 = -M00 + M02 # does not contain 00 OR has an even number of 0's

# E1 = eNFA({'s', 'A', 'B', 'C', 'f'}, {'0', '1'}, {('s', ''): {'A', 'f'}, ('A', '0'): {'B'}, ('B', '1'): {'C'}, ('C', ''): {'f'}, ('f', ''): {'s'}}, {'s'}, {'f'})
# EC1 = E1.subset_construction()

R1 = RegExpr2eNFA('(0+1)*.0.0.(0+1)*')
MR1 = R1.subset_construction().minimize()
printDFA(MR1)

# N = eNFA({'ab', 'cd'}, {'0', '1'}, {('ab', '0'): {'ab'}, ('ab', ''): {'cd'}}, {'ab'}, {'cd'})
# eNFA2RegExpr(N)