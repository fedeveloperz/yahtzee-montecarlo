"""Pruebas rapidas: ejecutar con  pytest -q  desde la raiz del repositorio."""
import math, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import yahtzee_mc as y


def test_puntuacion_categorias():
    assert y.score_hand("fullHouse", [3, 3, 3, 5, 5]) == 25
    assert y.score_hand("largeStraight", [1, 2, 3, 4, 5]) == 40
    assert y.score_hand("smallStraight", [1, 2, 3, 4, 4]) == 30
    assert y.score_hand("yahtzee", [6] * 5) == 50
    assert y.score_hand("threeOfKind", [2, 2, 2, 5, 6]) == 17
    assert y.score_hand("fours", [4, 4, 1, 2, 4]) == 12


def test_distribucion_dado_cargado():
    n = 200_000
    x = y.Dice(y.LOADED_6_30, seed=1).roll(n)
    assert abs(x.count(6) / n - 0.30) < 0.005          # Ley de los Grandes Numeros


def test_ultima_tirada_conserva_full():
    d = y.Dice(seed=1)
    m = y.elegir_mascara([3, 3, 3, 5, 5], 1, y.CATEGORIES_ORDER, d, y.Estrategia("montecarlo", 200, "voraz", True))
    assert all(m)                                       # el original devolvia [False]*5


def test_partida_completa_y_bono():
    s = y.play_game((y.Estrategia("frecuente", 0), y.Estrategia("frecuente", 0)), seed=3)
    for p in s.players:
        assert not p.available_categories()
        assert p.total == sum(p.scores.values()) + p.bonus
    assert s.stats.total_rolls < 26 * 15                # solo se cuentan dados realmente lanzados


def test_markov_vs_montecarlo():
    ex = y.p_yahtzee_exacto(y.FAIR)
    assert abs(ex - 0.04603) < 1e-4
    assert abs(y.p_yahtzee_mc(400_000, y.FAIR, seed=5) - ex) < 4 * math.sqrt(ex * (1 - ex) / 400_000)


def test_tres_iguales_exacto_vs_montecarlo():
    ex = y.tres_iguales_exacto(y.FAIR)
    m = y.tres_iguales_montecarlo(60_000, y.FAIR, seed=2)
    inf = sum(sum(ex[k]["pab"][ab] * max(ex[k]["detener"][ab], ex[k]["relanzar"]) for ab in ex[k]["pab"]) for k in y.FACES) / 6
    assert abs(m["informada"] - inf) < 0.2
    assert abs(m["p_yahtzee"] - 1 / 36) < 0.004


def test_dp_optimo_mayor_que_heuristica():
    assert y.esperanza_turno_optimo(y.FAIR) > 27
