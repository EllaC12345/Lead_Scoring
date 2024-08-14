# BUSINESS SCIENCE UNIVERSITY
# COURSE: DS4B 201-P PYTHON MACHINE LEARNING
# MODULE 1: BUSINESS UNDERSTANDING



#%%
# TEST CALCULATIONS ----
import email_lead_scoring as els
els.cost_calc_monthly_cost_table() 


els.cost_calc_monthly_cost_table() \
    .pipe(els.cost_total_unsub_cost)
    
els.cost_simulate_unsub_cost() 

els.cost_simulate_unsub_cost()\
    .pipe(els.cost_plot_simulated_unsub_costs)

# no need to do this if the init file is set up correctly by importing the submodules
#import email_lead_scoring.cost_calculations as cost
#cost.cost_calc_monthly_cost_table()



# %%
