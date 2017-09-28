import random
import genetic_functions
import copy
import pop_manager

class Chromosome(object):
    '''
    Encapsulates a chromosome
    '''

    def __init__(self, cfg):
        self.cfg = cfg
        self.init_method = self.cfg.init_method
        self.initial_max_depth = self.cfg.chromosome_initial_max_depth
        self.max_depth = self.cfg.chromosome_max_depth
        self.feature_number = self.cfg.feature_number
        self.constants_range = self.cfg.constants_range
        self.functions = self.cfg.genetic_functions
        self.dataset = self.cfg.dataset

        #  program is the chromosome tree
        self.program = []

        #self.generate()
        #print '[INFO]', 'Is valid?:{} Chromosome: {}'.format(self.validate(), self.program)

    def generate(self):
        self.program = self.build_program()
        return self

    def build_program(self):
        """
        Generates a new random chromosome
        """

        method = self.init_method
        if self.init_method == 'half/half':
            # method = random.choice(['grow', 'full'])
            if random.uniform(0.0, 1.0) < 0.4:
                method = 'grow'
            else:
                method = 'full'


        max_depth = random.randint(0, self.initial_max_depth - 1)

        # Start a program with a function to avoid degenerative programs
        #GeneticFunctions
        f = random.choice(self.functions.functions)
        program = [f]
        terminal_stack = [f['arity']]

        while len(terminal_stack) != 0:
            depth = len(terminal_stack)
            choice = self.feature_number + len(self.functions.functions)
            choice = random.randint(0, choice - 1)

            # Determine if we are adding a function or terminal
            if (depth < max_depth) and (method == 'full' or choice <= len(self.functions.functions)):
                f = random.choice(self.functions.functions)
                program.append(f)
                terminal_stack.append(f['arity'])
            else:
                # We need a terminal, add a variable or constant
                terminal = random.randint(0, self.feature_number)
                if terminal == self.feature_number:
                    terminal = random.uniform(self.constants_range[0], self.constants_range[1])

                program.append(terminal)
                terminal_stack[-1] -= 1
                while terminal_stack[-1] == 0:
                    terminal_stack.pop()
                    if len(terminal_stack) == 0:
                        return program
                    terminal_stack[-1] -= 1

        # We should never get here
        raise ValueError('Program cannot be created')

    def is_valid(self):
        """
        Validate if the chromosome is valid
        """
        if self.get_depth() > self.cfg.chromosome_max_depth:
            return False

        if self.get_length() == 1 and (isinstance(self.program[0], float) or isinstance(self.program[0], int)):
            return False

        terminals = [0]
        for node in self.program:
            if isinstance(node, dict):
                terminals.append(node['arity'])
            else:
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
        return terminals == [-1]

    def run_with_dataset(self):
        """
        Execute the chromosome using the dataset available
        :return: 
        """
        result_list = []

        # Check for single-node programs
        for var in self.dataset:
            node = self.program[0]
            if isinstance(node, float):
                result_list.append(node)
                continue
            if isinstance(node, int):
                result_list.append(var[node])
                continue

            apply_stack = []
            for node in self.program:
                if isinstance(node, dict):
                    apply_stack.append([node])
                else:
                    # Lazily evaluate later
                    apply_stack[-1].append(node)

                while len(apply_stack[-1]) == apply_stack[-1][0]['arity'] + 1:
                    # apply functions that have sufficient arguments
                    function = apply_stack[-1][0]

                    terminals = []
                    for t in apply_stack[-1][1:]:
                        if isinstance(t, float):
                            terminals.append(t)
                        else:
                            if isinstance(t, int):
                                #print 'var', var, 't', t
                                terminals.append(copy.deepcopy(var[t]))

                    # execute the terminals
                    intermediate_result = function['function'](*terminals)
                    #print function['name'], terminals, '=', intermediate_result
                    if len(apply_stack) != 1:
                        apply_stack.pop()
                        apply_stack[-1].append(intermediate_result)
                    else:
                        result_list.append(intermediate_result)
                        break

        return result_list

    def export_graphviz(self, fade_nodes=None):
        """
        Generates a graphviz string to visualize the chromosome 
        It can be seen on http://www.webgraphviz.com/
        """
        terminals = []
        if fade_nodes is None:
            fade_nodes = []
        output = "digraph program {\nnode [style=filled]"
        for i, node in enumerate(self.program):
            fill = "#cecece"
            if isinstance(node, dict):
                if i not in fade_nodes:
                    fill = "#136ed4"
                terminals.append([node['arity'], i])
                output += ('%d [label="%s", fillcolor="%s"] ;\n'
                           % (i, node['name'], fill))
            else:
                if i not in fade_nodes:
                    fill = "#60a6f6"
                if isinstance(node, int):
                    output += ('%d [label="%s%s", fillcolor="%s"] ;\n'
                               % (i, 'X', node, fill))
                else:
                    output += ('%d [label="%.3f", fillcolor="%s"] ;\n'
                               % (i, node, fill))
                if i == 0:
                    # A degenerative program of only one node
                    return output + "}"
                terminals[-1][0] -= 1
                terminals[-1].append(i)
                while terminals[-1][0] == 0:
                    output += '%d -> %d ;\n' % (terminals[-1][1],
                                                terminals[-1][-1])
                    terminals[-1].pop()
                    if len(terminals[-1]) == 2:
                        parent = terminals[-1][-1]
                        terminals.pop()
                        if len(terminals) == 0:
                            return output + "}"
                        terminals[-1].append(parent)
                        terminals[-1][0] -= 1

        # We should never get here
        return None

    def get_depth(self):
        """Calculates the maximum depth of the program tree."""
        terminals = [0]
        depth = 1
        for node in self.program:
            if isinstance(node, dict):
                terminals.append(node['arity'])
                depth = max(len(terminals), depth)
            else:
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
        return depth - 1

    def get_length(self):
        """Calculates the number of functions and terminals in the program."""
        return len(self.program)

    def subtree_mutation(self):
        """Perform the subtree mutation operation on the program.

        Subtree mutation selects a random subtree from the embedded program to
        be replaced. A donor subtree is generated at random and this is
        inserted into the original parent to form an offspring. This
        implementation uses the "headless chicken" method where the donor
        subtree is grown using the initialization methods and a subtree of it
        is selected to be donated to the parent.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.
        """
        # Build a new naive program
        chicken = Chromosome(self.cfg).generate()
        # Do subtree mutation via the headless chicken method!

        mutated_list = pop_manager.PopManager.crossover(self, chicken)
        self.from_list(mutated_list)
        return mutated_list

    def hoist_mutation(self):
        """Perform the hoist mutation operation on the program.

        Hoist mutation selects a random subtree from the embedded program to
        be replaced. A random subtree of that subtree is then selected and this
        is 'hoisted' into the original subtrees location to form an offspring.
        This method helps to control bloat.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.
        """
        # Get a subtree to replace
        start, end = self.get_subtree()
        subtree = self.program[start:end]
        #subtree_chrom = Chromosome(self.cfg).from_list(subtree)

        # Get a subtree of the subtree to hoist
        # sub_start, sub_end = subtree_chrom.get_subtree()
        # hoist = subtree[sub_start:sub_end]
        #
        # print 'start, end', start, end
        #
        # str_list = []
        # for i in xrange(len(self.program[:start])):
        #     if isinstance(self.program[:start][i], dict):
        #         str_list.append(self.program[:start][i]['name'])
        #     else:
        #         str_list.append(self.program[:start][i])
        # print '1:', str_list
        #
        # str_list = []
        # for i in xrange(len(hoist)):
        #     if isinstance(hoist[i], dict):
        #         str_list.append(hoist[i]['name'])
        #     else:
        #         str_list.append(hoist[i])
        # print 'hoist', str_list
        #
        # str_list = []
        # for i in xrange(len(self.program[end:])):
        #     if isinstance(self.program[end:][i], dict):
        #         str_list.append(self.program[end:][i]['name'])
        #     else:
        #         str_list.append(self.program[end:][i])
        # print '2:', str_list
        #
        # mutated_list = self.program[:start] + hoist + self.program[end:]
        # str_list = []
        # for i in xrange(len(mutated_list)):
        #     if isinstance(mutated_list[i], dict):
        #         str_list.append(mutated_list[i]['name'])
        #     else:
        #         str_list.append(mutated_list[i])
        # print '3:', str_list
        # #print '3:', mutated_list

        #self.from_list(mutated_list)
        self.from_list(subtree)
        return subtree

    def point_mutation(self):
        """
        Point mutation operation on the chromosome.
        Point mutation selects random nodes from the embedded program to be
        replaced. Terminals are replaced by other terminals and functions are
        replaced by other functions that require the same number of arguments
        as the original node. The resulting tree forms an offspring.
        """

        for i in xrange(len(self.program)):
            prob = random.uniform(0.0, 1.0)
            #print 'prob', prob, ' self.cfg.p_mutation_point',  self.cfg.p_mutation_point
            if prob < self.cfg.p_mutation_point:
                g = self.program[i]
                g_old = g

                if isinstance(g, dict):
                    # if is function
                    g_arity = g['arity']
                    a_list = [e for e in self.functions.functions if e['arity'] == g_arity]
                    g = random.choice(a_list)
                    #print '[DEBUG]', 'Function GArity:{} g_old:{} g_new:{}'.format(g_arity, g_old['name'], g['name'])

                elif isinstance(self.program[i], float):
                    # if is constant
                    g = random.uniform(self.constants_range[0], self.constants_range[1])
                    #print '[DEBUG]', 'Constant g_old:{} g_new:{}'.format(g_old, g)

                else:
                    # if is variable
                    g = random.randint(0, self.feature_number - 1)
                    #print '[DEBUG]', 'Variable g_old:{} g_new:{}'.format(g_old, g)

                self.program[i] = g
        return self.program

    def get_subtree(self):
        """
        Get a random subtree
        Use the prob 90% for function and 10% for leaf
        Set the probabilities for every node
        Then normalize and select by cdf of the normalized data
        :return start, end : tuple of two ints
        """

        #print 'get subtree'

        prob_list = []
        for i in xrange(len(self.program)):
            g = self.program[i]
            if isinstance(g, dict):
                prob_list.append(0.9)
            else:
                prob_list.append(0.1)

        s = sum(prob_list)
        norm = [float(i) / s for i in prob_list]

        prob = random.random()
        cdf = 0
        selected_index = -1
        for i in xrange(len(norm)):
            v = norm[i]
            cdf += v
            #print 'fx', cdf, prob, 'index', i
            if prob <= cdf:
                selected_index = i
                break

        start = selected_index
        stack = 1
        end = start

        #print 'start', start, 'stack', stack, 'end', end

        while stack > end - start:
            node = self.program[end]
            if isinstance(node, dict):
                stack += node['arity']
            end += 1
            #print 'start', start, 'stack', stack, 'end', end , 'node', node

        return start, end

    def from_list(self, program):
        self.program = []
        for node in program:
            if isinstance(node, str):
                funct = self.functions.get_function(node)
                self.program.append(funct)
            else:
                self.program.append(node)
        return self

    def to_list(self):
        rep = []
        for node in self.program:
            if isinstance(node, dict):
                rep.append(node['name'])
            else:
                rep.append(node)
        return rep

    def clone(self):
        #print 'original_program', self.program
        #print 'before', self.to_list()
        chrom = Chromosome(self.cfg).from_list(self.to_list())
        #print 'after', self.to_list()
        return chrom
