# Yahtzee de 2 jugadores y método de Montecarlo — Actividad didáctica 2-M1

Simulación en **Python** del Yahtzee clásico (5 dados, hasta 3 lanzamientos por turno, 13 categorías, 2 jugadores) donde una IA decide qué dados conservar estimando por **Montecarlo** el valor esperado de cada opción. Se verifica la **Ley de los Grandes Números** y se compara con soluciones analíticas. Juego de referencia: <https://solitaired.com/yahtzee>.

Metodología **USA → MODIFICA → CREA** (+ DEBATE), sobre el notebook base `simulacion_01`:

| Fase | Contenido |
|---|---|
| **USA** | Código base sin cambios (`src/simulacion_original.py`), diagrama de flujo, partida completa, convergencia con N = 10, 100, 1.000, 10.000 y 100.000 frente al valor exacto |
| **MODIFICA** | Corrección de la última tirada (el original relanzaba siempre los 5 dados), conteo correcto de dados lanzados, dado parametrizable (dado cargado: 6 con 30 %), bono de 35, categoría por valor esperado, números aleatorios comunes, memoización; cada mejora medida con IC 95 % |
| **CREA** | Tres dados iguales: ¿lanzar los otros dos o detenerse? Valor esperado exacto y por Montecarlo, y mapa de decisión |
| **DEBATE** | Montecarlo frente a programación dinámica (óptimo de un turno) y cadena de Markov (P(Yahtzee)) |

## Estructura

```
notebooks/Yahtzee_Montecarlo_USA_MODIFICA_CREA.ipynb   notebook principal, ya ejecutado
src/simulacion_original.py   código base entregado (sin modificar)
src/yahtzee_mc.py            motor modificado (módulo reutilizable)
tests/test_yahtzee_mc.py     pruebas (pytest)
resultados/                  figuras (.png) y resumen.json con todas las cifras
docs/                        documento técnico (.docx) y guion del video
```

## Cómo ejecutar

```bash
pip install numpy pandas matplotlib graphviz jupyter   # graphviz también requiere el programa `dot`
jupyter notebook notebooks/Yahtzee_Montecarlo_USA_MODIFICA_CREA.ipynb   # Ejecutar todo (~7 min)
pytest -q tests                                                         # 7 pruebas
```

En **Google Colab**: subir el notebook y ejecutar todo (genera `simulacion_original.py` y `yahtzee_mc.py` con `%%writefile`). Con `RAPIDO = True` tarda ~2 min. Semilla fija: `SEMILLA = 2026`.

## Resultados principales

| | Resultado |
|---|---|
| Convergencia (EV de `[3,3,5,2,6]`, exacto 17,301) | error medio 0,91 → 0,093 → 0,0065 con N = 10, 1.000, 100.000 (∝ 1/√N) |
| P(Yahtzee) en un turno (exacta 0,0460) | 0,000 / 0,041 / 0,0455 / 0,0460 con N = 10 / 1.000 / 10⁵ / 10⁶ |
| Puntaje medio por jugador | original ≈ 129 → +M1 ≈ 207 → +M5 ≈ 218 → +M6 ≈ 221 |
| Dado cargado (6 con 30 %) | P(Yahtzee) 0,046 → 0,113; puntaje medio ≈ 221 → ≈ 248 |
| CREA: tres dados iguales | detenerse = relanzar a ciegas = 19,47; política informada = 21,34 |
| DEBATE: turno óptimo exacto 28,46 | Montecarlo (1.000 sims) 28,29; heurística simple 21,98 |

## Referencias
- Metropolis, N. y Ulam, S. (1949). The Monte Carlo method. *JASA, 44*(247), 335-341.
- Grinstead, C. M. y Snell, J. L. (1997). *Introduction to Probability* (2.ª ed.). AMS.
- Ross, S. M. (2013). *Simulation* (5.ª ed.). Academic Press.
- Glenn, J. (2006). An optimal strategy for Yahtzee. Loyola College in Maryland.
- Lee, I., Martin, F. y Apone, K. (2014). Integrating computational thinking across the K-8 curriculum. *ACM Inroads, 5*(4), 64-71.
