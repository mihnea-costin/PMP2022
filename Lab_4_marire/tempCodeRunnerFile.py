q = pm.quantiles(trace['total_time'], q=[0.95])

# # Verificare conditie
# if q[0.95] <= max_time:
#     print(f"Valoarea maxima pentru alpha este {alpha_max:.2f} minute")
# else:
#     print("Nu exista o valoare maxima pentru alpha astfel incat toti clientii care nu cumpara sa nu stea in magazin mai mult de 15 minute, cu o probabilitate de 95%")
