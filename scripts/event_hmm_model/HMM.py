# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 01:29:18 2022

@author: gianl
"""

import os
import numpy as np
import pandas as pd
from dataclasses import dataclass

from ATMANpy.funcs.HMM.utils import calculate_emission_densities

np.seterr(divide = 'ignore')

def read_grammar(grammarStr):
    from ATMANpy.funcs.util_funcs import get_folder_struct
    grammarDir = os.path.join(get_folder_struct('ATMANpy')['auxil'], 'backus_naur_grammars')
    grammarFile = grammarStr + '.txt'
    grammarPath = os.path.join(grammarDir, grammarFile)
    
    with open(grammarPath) as f:
        lines = [line.strip() for line in f]

    grammarDict = {}
    for l in lines:
        symbol, expression = l.split('::=')
        symbol, expression = symbol.strip(' '), expression.strip(' ')
        grammarDict[symbol] = expression
    
    return grammarDict

GRAMMAR_STR = 'serve_fault'
GRAMMAR = read_grammar(GRAMMAR_STR)

from ATMANpy.funcs.HMM.utils import read_hmm_params, read_train_test
VIDEO_ID = 'pqxAmUWIoKc'
MODEL_NAME = 'embedded_cyc3_nrel85'
HMMs = read_hmm_params(MODEL_NAME, VIDEO_ID)

train_data, test_data = read_train_test(MODEL_NAME, VIDEO_ID)
randseqIdx = np.random.choice(test_data.IndexSF.unique())
randSeq = test_data[test_data.IndexSF == randseqIdx]
WORDS = randSeq.drop(columns = ['Event', 'IndexSF'])

#%%

WLRDICT = {key: {} for key in WORDS.index}

@dataclass
class HMM:
    A: np.array
    avg: pd.DataFrame
    cov: pd.DataFrame
    
    def __post_init__(self):
        self.log_A = np.log(self.A)
        self.log_A_init = self.log_A[0, 1:-1]
        self.log_A_emit = self.log_A[1:-1, 1:-1]
        self.log_A_exit = self.log_A[1:-1, -1]
        self.states = len(self.avg)
        
        self.isTerminal = True ### word models are terminal symbols in our syntax!
        
        self.entry_token_value = np.log(1)
        self.entry_token_id = -1
        
        self.token_values = np.log(np.zeros(self.states))
        self.token_ids = np.ones(self.states, dtype = int) * -1
        
        self.exit_token_value = np.log(0)
        self.exit_token_id = -1

    def init_log_density(self, words, density_floor = 10 ** -100):
        densFrm = calculate_emission_densities(words, self.avg, self.cov, 
                                               density_floor = density_floor)
        self.log_densFrm = np.log(densFrm)
    
    def get_density_at(self, t):
        return self.log_densFrm.loc[t].values

    def set_entry_token(self, token_value, token_id):
        self.entry_token_value = token_value
        self.entry_token_id = token_id
    
    def get_exit_token(self):
        return self.exit_token_value, self.exit_token_id
        
    def step_tokens_into_hmm(self, t):
        ### token value: scalar
        ### token id: scalar
        ### log_density: vector S
        
        log_density = self.get_density_at(t)
        
        ####### TRANSITION INTO HMM ###########
        entry_token_values = self.entry_token_value + self.log_A_init + log_density
        entry_token_ids = [self.entry_token_id] * self.states
            
        return entry_token_values, entry_token_ids
        
    def step_tokens_within_hmm(self, t):
        ### log_density: vector S
        
        log_density = self.get_density_at(t)
        
        ####### TRANSITION WITHIN HMM #########

        ### broadcast these to SxS
        token_values_brd = self.token_values[:, np.newaxis] #### shape: S_from, S_to
        log_dens_values_brd = log_density[np.newaxis, :] ### shape: S_from, S_to

        ### calculate token values
        emit_token_values_brd = token_values_brd + self.log_A_emit + log_dens_values_brd
        emit_token_values = emit_token_values_brd.max(axis = 0) ### in every column, find maximum
        emit_token_indices = emit_token_values_brd.argmax(axis = 0)
        emit_token_ids = self.token_ids[emit_token_indices]
        
        return emit_token_values, emit_token_ids
    
    def step_tokens_out_of_hmm(self):
        ###### TRANSITION OUT OF HMM ###########
        
        ### calculate token values
        exit_token_values = self.token_values + self.log_A_exit
        exit_token_value = exit_token_values.max(axis = 0) ### in every column, find maximum
        exit_token_index = exit_token_values.argmax(axis = 0)
        exit_token_ids = self.token_ids[exit_token_index]

        return exit_token_value, exit_token_ids
    
    def set_step_tokens(self, t):
        entry_token_values, entry_token_ids = self.step_tokens_into_hmm(t) 
        
        emit_token_values, emit_token_ids = self.step_tokens_within_hmm(t) 
        
        ### COMPARE TRANSITION INTO AND WITHIN
        entry_likelier = entry_token_values > emit_token_values ### by state, is it likelier
                                                                ### to come from outside?
        
        new_token_values = np.where(entry_likelier, entry_token_values, emit_token_values)
        new_token_ids = np.where(entry_likelier, entry_token_ids, emit_token_ids)

        ### Set emission token values
        self.token_values = new_token_values
        self.token_ids = new_token_ids
            
        ### Set exit token values
        exit_token_value, exit_token_ids = self.step_tokens_out_of_hmm()
        self.exit_token_value = exit_token_value
        self.exit_token_id = exit_token_ids
        
        ### Re-Set entry token values
        self.entry_token_value = np.log(0)
        self.entry_token_id = -1
    
    
    
@dataclass
class Link:
    prec: list ### list of preceeding models
    succ: list ### list of succeeding models
    
    def __post_init__(self):
        self.token_value = np.log(0)
        self.token_id = -1 
        self.emitting_model = None
        self.WLRs = WLRDICT  ### assign a common storage for a word link records
        
    # def set_WLRs(self, wlrDict):
    #     self.wlrDict = wlrDict ### assign a common storage for a word link records
    
    # def set_WLRs(self, WLRs):
    #     ### list of word link records, coming from the superordinate model shell
    #     self.WLRs = WLRs
    
    def set_filter_tokens(self, model, model_name):
        model_exit_token_value, model_exit_token_id = model.get_exit_token()
        if model_exit_token_value > self.token_value:
            self.token_value = model_exit_token_value
            self.token_id = model_exit_token_id
            self.emitting_model = model_name
    
    def record_decisions(self, t):
        
        ### make new wlr entry
        wlrs_at_t = self.WLRs[t]
        if wlrs_at_t == {}:
            new_key = 0
        else:
            new_key = max(wlrs_at_t) + 1 ### get the maximum key, add one
        
        new_wlr_entry = {'token_value': self.token_value, 
                         'token_id': self.token_id, 
                         't': t, 
                         'emitting_model': self.emitting_model}
        wlrs_at_t[new_key] = new_wlr_entry
        
        ### replace token id with wlr pointer
        self.token_id = str(t) + '_' + str(new_key)
        
    
    def push_token_to_entry(self, t):
        for shell in self.succ:
            shell.inject_token(self.token_value, self.token_id)
            model = shell.model
            if not model is None: ### only if the model shell has been activated.
                if not model.isTerminal:
                    model.propagate_entry_tokens(t)
                else:
                    model.set_step_tokens(t) ### when pushing, we also move ahead terminal models
                    
        self.token_value = np.log(0)
        self.token_id = -1 
                
            
    def pull_token_from_exit(self, t):
        for shell in self.prec:
            model, name = shell.model, shell.name
            if not model is None: ### only if the model shell has been activated.
                if not model.isTerminal:
                    model.propagate_exit_tokens(t)
                self.set_filter_tokens(model, name)
        if self.emitting_model is not None: ### if the link has been reached with positive prob
            self.record_decisions(t)
    
@dataclass
class ModelShell:
    #### NEED A METACLASS TO CREATE MODEL SHELLS!
    # 1) pass in a grammar
    # 2) pass in a set of hmm parameters A, avg, cov
    
    name: str
    num: int
    
    def __post_init__(self):
        self.model = None # SubSyntaxModel | HMM | None

    def inject_token(self, token_value, token_id):
        if self.model is None: ### 'wakes up' the shell if a pos. prob. token is passed in
            if token_value > np.log(0):
                if self.name.startswith('<') and self.name.endswith('>'): ### SSM
                    symbol = self.name.strip('<>')
                    expression = GRAMMAR[symbol]
                    self.model = SubSyntaxModel.from_grammar_rule(expression)
                else: ### HMM
                    hmmDict = HMMs[self.name]
                    self.model = HMM(**hmmDict)
                    self.model.init_log_density(WORDS)
        
        if self.model is not None: ### once we have a model, set entry tokens inside them
            self.model.set_entry_token(token_value, token_id) 
        
    def propagate_entry_tokens(self, t):
        self.model.propagate_entry_tokens(t)
            
    def propagate_exit_tokens(self, t):
        self.model.propagate_exit_tokens(t)
              
                
@dataclass
class BinaryLink:
    ### a link connecting one model to another model
    prec: ModelShell
    succ: ModelShell
  
    def reverse(self):
        return BinaryLink(self.succ, self.prec)
  
    def is_entry(self):
        return self.prec is None
    def is_exit(self):
        return self.succ is None
  
@dataclass
class HalfLink:  
    ### a link connecting one model with many other models
    prec: ModelShell
    succ: list
    
      
def flatten(someList):
    return [el for inList in someList for el in inList]
     
@dataclass
class SubSyntaxModel:
    entry_link: Link
    intm_links: list
    exit_link: Link
    
    def __post_init__(self):
        self.isTerminal = False ### sub syntax models consist of other sub syntax models or HMMs

    @staticmethod
    def from_grammar_rule(ruleStr):
        possible_sequences = [elem.strip(' ') for elem in ruleStr.split('|')]
        
        ### first pass over sequence strings, to instantiate unique models
        instantiated_models = [] ### master list of instantiated models
        list_of_sequence_models = [] ### this list contains lists of references to instantiated models
        
        for sequence in possible_sequences:
            seqList = sequence.split(' ')
            
            sequence_models = []
            for model_name in seqList:
                prev_instances = [model.name == model_name for model in sequence_models]
                model_num = sum(prev_instances)
                
                new_model = ModelShell(model_name, model_num)
                
                if not new_model in instantiated_models: ### don't duplicate already instantiated models
                    instantiated_models.append(new_model)
                    sequence_models.append(new_model)
                else:
                    old_model = [mod for mod in instantiated_models if mod == new_model][0]
                    sequence_models.append(old_model)
                
            list_of_sequence_models.append(sequence_models)
                
           
        ### second pass over sequence models, to generate all binary links between model pairs
        binary_links = []
        for sequence_models in list_of_sequence_models:
            
            ### initalize the entry link -> no previous model
            prev_model = None
            
            ### go through models in sequence
            for model in sequence_models:
                
                binLink = BinaryLink(prev_model, model)
                if not binLink in binary_links:
                    binary_links.append(binLink)
                
                prev_model = model
         
            ### generate exit link
            binLink = BinaryLink(prev_model, None)
            if not binLink in binary_links:
                binary_links.append(binLink)
         
        ### check for dangerous two way links
        for binLink in binary_links:
            if binLink.reverse() in binary_links:
                if not binLink.is_entry() and not binLink.is_exit():
                    print(f'Warning: {ruleStr} implies two way binary link: {binLink}')
         
        ### third pass over preceeding models of binary links, to create half links
        prec_models = []
        for binLink in binary_links:
            precMod = binLink.prec
            if not precMod in prec_models:
                prec_models.append(precMod)
        
        # group binary links by preceding model
        half_links = []
        for precMod in prec_models:
            same_prec_links = [link for link in binary_links if link.prec == precMod]
            succList = [link.succ for link in same_prec_links if not link.succ is None] ## None succ -> exit link
            
            halfLink = HalfLink(precMod, succList)
            half_links.append(halfLink)
        
        #### fourth pass over succeeding model lists of half links, to create full links
        succ_model_lists = []
        for halfLink in half_links:
            succMods = halfLink.succ
            if not succMods in succ_model_lists:
                succ_model_lists.append(succMods)
         
        # group half links by succeeding models
        links = []
        for succMods in succ_model_lists:
            same_succ_links = [link for link in half_links if link.succ == succMods]
            precMods = [link.prec for link in same_succ_links if not link.prec is None] ## None succ -> entry link
            
            fullLink = Link(precMods, succMods) ### pass in the superordinate list of WLRs
            # fullLink.set_WLRs(WLRs)
            links.append(fullLink)   
            
        ### split up links into entry, intermediate, and exit link
        entry_link = [l for l in links if l.prec == []][0]
        exit_link = [l for l in links if l.succ == []][0]
        intm_links = [l for l in links if l not in [entry_link, exit_link]]
            
        return SubSyntaxModel(entry_link, intm_links, exit_link)
            
            
    def set_entry_token(self, token_value, token_id):
        self.entry_link.token_value = token_value
        self.entry_link.token_id = token_id

    def get_exit_token(self):
        return self.exit_link.token_value, self.exit_link.token_id

    def propagate_entry_tokens(self, t):
        print('TO ENTRY:')
        for link in [self.entry_link, *self.intm_links, self.exit_link]:
            print(link)
            link.push_token_to_entry(t)
            
    def propagate_exit_tokens(self, t):
        print('FROM EXIT:')
        for link in [self.entry_link, *self.intm_links, self.exit_link]:
            print(link)
            link.pull_token_from_exit(t)
 
    
def work_back_WLR_links(modShell):
    
    last_token_id = modShell.model.exit_link.token_id
    # WLRs = modShell.WLRs
    
    word_sequence = []
    
    while last_token_id not in [-1, 0]:
        t, num = last_token_id.split('_') ### a token id consists of time and an identifying number
        t, num = int(t), int(num)
        
        WLR = WLRDICT[t][num]
        exiting_model, period_exited = WLR['emitting_model'], WLR['t']
    
        if not (exiting_model.startswith('<') and exiting_model.endswith('>')):
            word_sequence.append((exiting_model, period_exited))
    
        last_token_id = WLR['token_id']

    return word_sequence


start = min(WORDS.index)
stop = max(WORDS.index)


allMod = ModelShell('<all>', 0)
allMod.inject_token(np.log(1), 0)
for t in WORDS.index:
    allMod.propagate_entry_tokens(t)
    allMod.propagate_exit_tokens(t)

inferred_sequence = work_back_WLR_links(allMod)

from ATMANpy.funcs.util_funcs import get_video        
from ATMANpy.funcs.cv_funcs import show_video, get_frame, show

cap = get_video(VIDEO_ID)
show_video(cap, start_frame = start, stop_frame = stop)
     
show(get_frame(cap, 89201))  

modShell = allMod 
