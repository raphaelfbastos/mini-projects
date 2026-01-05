from automata import *

# DFAs
M1 = DFA({'A', 'B', 'C'}, {'0', '1'}, {('A', '0'): 'B', ('A', '1'): 'A', ('B', '0'): 'C', ('B', '1'): 'B', ('C', '0'): 'C', ('C', '1'): 'C'}, 'A', {'C'})
M2 = DFA({'S', 'A', 'B', 'C', 'D'}, {'0'}, {('S', '0'): 'A', ('A', '0'): 'B', ('B', '0'): 'C', ('C', '0'): 'D', ('D', '0'): 'A'}, 'S', {'B', 'D'})
print('=======================DFAs=======================')
print('\n~~~~~~~M1~~~~~~~')
print(M1)
print('\n~~~~~~~M2~~~~~~~')
print(M2)

# DFA minimization
M3 = DFA({'0', '1', '2', '3', '4', '5', '6', '7'}, {'a', 'b'}, {('0', 'a'): '1', ('0', 'b'): '2', ('1', 'a'): '3', ('1', 'b'): '4', ('2', 'a'): '4', ('2', 'b'): '3', ('3', 'a'): '5', ('3', 'b'): '5', ('4', 'a'): '5', ('4', 'b'): '5', ('5', 'a'): '5', ('5', 'b'): '5', ('6', 'a'): '3', ('6', 'b'): '5', ('7', 'a'): '5', ('7', 'b'): '4'}, '0', {'1', '2', '5'})
Q1 = M3.minimize()
print('\n=================DFA minimization=================')
print('\n~~~~~~~M3~~~~~~~')
print(M3)
print('\n~~~~~~~Q1~~~~~~~')
print(Q1)

# NFA and subset construction
N1 = NFA({'A', 'B'}, {'a', 'b'}, {('A', 'a'): {'A'}, ('A', 'b'): {'B'}}, {'A'}, {'B'}) # has any number of a's followed by a single b
C1 = N1.subset_construction()
print('\n===========NFA and subset construction============')
print('\n~~~~~~~N1~~~~~~~')
print(N1)
print('\n~~~~~~~C1~~~~~~~')
print(C1)

# Product Machine construction
M00 = DFA({'A', 'B', 'C'}, {'0', '1'}, {('A', '0'): 'B', ('A', '1'): 'A', ('B', '0'): 'C', ('B', '1'): 'A', ('C', '0'): 'C', ('C', '1'): 'C'}, 'A', {'C'}) # contains 00
M02 = DFA({'A', 'B'}, {'0', '1'}, {('A', '0'): 'B', ('A', '1'): 'A', ('B', '0'): 'A', ('B', '1'): 'B'}, 'A', {'A'}) # has an even number of 0's
P1 = -M00 * M02 # does not contain 00 AND has an even number of 0's
P2 = -M00 + M02 # does not contain 00 OR has an even number of 0's
print('\n===========Product Machine construction===========')
print('\n~~~~~~M00~~~~~~~')
print(M00)
print('\n~~~~~~M02~~~~~~~')
print(M02)
print('\n~~~~~~~P1~~~~~~~')
print(P1)
print('\n~~~~~~~P2~~~~~~~')
print(P2)

# eNFA and modified subset construction
E1 = eNFA({'s', 'A', 'B', 'C', 'f'}, {'0', '1'}, {('s', ''): {'A', 'f'}, ('A', '0'): {'B'}, ('B', '1'): {'C'}, ('C', ''): {'f'}, ('f', ''): {'s'}}, {'s'}, {'f'})
C2 = E1.subset_construction()
print('\n======eNFA and modified subset construction=======')
print('\n~~~~~~~E1~~~~~~~')
print(E1)
print('\n~~~~~~~C2~~~~~~~')
print(C2)

# Regular Expression to eNFA to minimized DFA
R1 = RegExpr2eNFA('(0+1)*.0.0.(0+1)*')
M4 = R1.subset_construction().minimize()
print('\n===Regular Expression to eNFA to minimized DFA====')
print('\n~~~~~~~R1~~~~~~~')
print(R1)
print('\n~~~~~~~M4~~~~~~~')
print(M4)

# Word filter using Regular Expression to eNFA to minimized DFA for acceptance of words in list.
# Words accepted by the DFA are words that should be filtered out.
WORDS = ['badword1', 'badword2', 'badword3', 'BIGBADWORD']
R = '+'.join(['.'.join(list(w)) for w in WORDS])
N = RegExpr2eNFA(R)
M = N.subset_construction().minimize()
print('\n===================Word filter====================')
print('\n~~~~~~~M~~~~~~~~')
print(M)
print()

SENTENCE = 'I often say goodword followed by littlebadword, sometimes badword1, occasionally badword2, and rarely badword3. Never BIGBADWORD.'
FILTERED = ' '.join(['*'*len(w) if M.accepts(w) else w for w in SENTENCE.replace(',', ' ,').replace('.', ' .').split()]).replace(' ,', ',').replace(' .', '.')
print('SENTENCE:', SENTENCE)
print('FILTERED:', FILTERED)
