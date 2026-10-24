"""
yahtzee_mc.py  -  Yahtzee de 2 jugadores con decisiones por Montecarlo.

Es la version MODIFICADA (etapa MODIFICA) del codigo base 'simulacion_01':
mantiene su estructura (dados, puntuacion, PlayerState, GameState, mascaras de
bloqueo, simulacion Montecarlo del valor esperado) y le agrega:

  M1  Correccion: en la ultima tirada el original relanzaba SIEMPRE los 5 dados.
  M2  Correccion: el histograma de caras contaba dos veces los dados bloqueados.
  M3  Distribucion del dado parametrizable (justo, cargado, ...).
  M4  Bono de 35 puntos por suma superior >= 63 (regla clasica, faltaba).
  M5  Eleccion de categoria por 'puntaje - valor esperado de la categoria'
      (el original elegia el puntaje maximo, desperdiciando Yahtzee/Escaleras).
  M6  Numeros aleatorios comunes (CRN) entre mascaras: menos varianza y mas rapido.
  M7  Memoizacion de la puntuacion (252 manos posibles).
  M8  Semilla por partida (reproducibilidad) y estrategias por jugador.
  M9  Solucion exacta por programacion dinamica para validar a Montecarlo.
"""
import itertools
import math
import random
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

# ----------------------------------------------------------------------------
# Constantes y distribuciones de probabilidad del dado
# ----------------------------------------------------------------------------
FACES = (1, 2, 3, 4, 5, 6)
FAIR = (1/6,) * 6                      # dado justo (distribucion uniforme)
LOADED_6_30 = (0.14,) * 5 + (0.30,)    # dado cargado: el 6 sale con 30 %

CATEGORIES_ORDER = [
    "ones", "twos", "threes", "fours", "fives", "sixes",
    "threeOfKind", "fourOfKind", "fullHouse", "smallStraight",
    "largeStraight", "yahtzee", "chance",
]
UPPER = CATEGORIES_ORDER[:6]
BONUS_THRESHOLD, BONUS_POINTS = 63, 35


class Dice:
    """Generador de dados con distribucion `probs` (transformada inversa discreta)."""

    def __init__(self, probs: Sequence[float] = FAIR, seed: Optional[int] = None):
        if abs(sum(probs) - 1) > 1e-9 or len(probs) != 6:
            raise ValueError("probs debe tener 6 valores que sumen 1")
        self.probs = tuple(probs)
        self.cum = list(itertools.accumulate(probs))
        self.rng = random.Random(seed)

    def roll(self, n: int = 1) -> List[int]:
        return self.rng.choices(FACES, cum_weights=self.cum, k=n)


# ----------------------------------------------------------------------------
# Puntuacion (misma logica del original; memoizada sobre la mano ordenada)
# ----------------------------------------------------------------------------
def has_sequence(values: Sequence[int], needed_len: int) -> bool:
    uniq = sorted(set(values))
    if not uniq:
        return False
    longest = curr = 1
    for i in range(1, len(uniq)):
        if uniq[i] == uniq[i - 1] + 1:
            curr += 1
            longest = max(longest, curr)
        else:
            curr = 1
    return longest >= needed_len


@lru_cache(maxsize=None)
def _score_sorted(category: str, dice: Tuple[int, ...]) -> int:
    counts = Counter(dice)
    freqs = list(counts.values())
    total = sum(dice)
    if category in UPPER:
        num = UPPER.index(category) + 1
        return counts.get(num, 0) * num
    if category == "threeOfKind":
        return total if any(f >= 3 for f in freqs) else 0
    if category == "fourOfKind":
        return total if any(f >= 4 for f in freqs) else 0
    if category == "fullHouse":
        return 25 if (2 in freqs and 3 in freqs) else 0
    if category == "smallStraight":
        return 30 if has_sequence(dice, 4) else 0
    if category == "largeStraight":
        return 40 if has_sequence(dice, 5) else 0
    if category == "yahtzee":
        return 50 if 5 in freqs else 0
    if category == "chance":
        return total
    return 0


def score_hand(category: str, dice: Sequence[int]) -> int:
    return _score_sorted(category, tuple(sorted(dice)))


def best_category_score(dice, available) -> Tuple[str, int]:
    """Seleccion VORAZ del original: la categoria con mayor puntaje bruto."""
    best_cat, best = None, -1
    for cat in available:
        sc = score_hand(cat, dice)
        if sc > best:
            best, best_cat = sc, cat
    return best_cat, best


# ----------------------------------------------------------------------------
# Valor esperado de cada categoria (referencia para M5), estimado por Montecarlo
# ----------------------------------------------------------------------------
def _mascara_frecuente(dados: Sequence[int]) -> List[bool]:
    """Conserva la cara mas repetida (en empate, la mayor)."""
    c = Counter(dados)
    cara = max(c, key=lambda k: (c[k], k))
    return [d == cara for d in dados]


@lru_cache(maxsize=None)
def baseline_categorias(probs: Tuple[float, ...] = FAIR, n_turnos: int = 20000,
                        seed: int = 12345) -> Dict[str, float]:
    """E[puntaje de cada categoria] en un turno jugado con la heuristica 'cara mas
    frecuente'. Es una estimacion Montecarlo (Ley de los Grandes Numeros)."""
    dice = Dice(probs, seed)
    acc = {c: 0.0 for c in CATEGORIES_ORDER}
    for _ in range(n_turnos):
        d = dice.roll(5)
        for _r in range(2):
            m = _mascara_frecuente(d)
            for i in range(5):
                if not m[i]:
                    d[i] = dice.roll(1)[0]
        for c in CATEGORIES_ORDER:
            acc[c] += score_hand(c, d)
    return {c: v / n_turnos for c, v in acc.items()}


# ----------------------------------------------------------------------------
# Estrategia de un jugador
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Estrategia:
    tipo: str = "montecarlo"          # 'montecarlo' | 'frecuente'
    sims_por_mascara: int = 400       # SIMS_PER_MASK del original
    seleccion: str = "ajustada"       # 'voraz' (original) | 'ajustada' (M5)
    fix_ultima_tirada: bool = True    # False = comportamiento del original (M1)
    crn: bool = True                  # numeros aleatorios comunes (M6)


ESTRATEGIA_ORIGINAL = Estrategia("montecarlo", 400, "voraz", False, False)


def all_keep_masks() -> List[List[bool]]:
    return [[(m >> i) & 1 == 1 for i in range(5)] for m in range(32)]


MASKS = all_keep_masks()


def _funcion_valor(disponibles: List[str], seleccion: str, base: Dict[str, float]):
    """value(mano_ordenada) = mejor 'puntaje' alcanzable con las categorias abiertas."""
    cache: Dict[Tuple[int, ...], float] = {}
    if seleccion == "voraz":
        def f(t):
            r = cache.get(t)
            if r is None:
                r = cache[t] = max(_score_sorted(c, t) for c in disponibles)
            return r
    else:
        def f(t):
            r = cache.get(t)
            if r is None:
                r = cache[t] = max(_score_sorted(c, t) - base[c] for c in disponibles)
            return r
    return f


def elegir_mascara(valores: List[int], tiradas_restantes: int, disponibles: List[str],
                   dice: Dice, est: Estrategia) -> List[bool]:
    """Elige que dados conservar. Montecarlo de 1 paso: para cada mascara se relanzan
    los dados libres una vez y se promedia el valor de la mano resultante."""
    if est.tipo == "frecuente":
        return _mascara_frecuente(valores)
    base = baseline_categorias(dice.probs)
    value = _funcion_valor(disponibles, est.seleccion, base)
    stand = value(tuple(sorted(valores)))
    if not est.fix_ultima_tirada and tiradas_restantes == 1:
        # Defecto del original: con 0 tiradas posteriores todas las EV empatan y
        # gana la primera mascara (todo False) => relanza los 5 dados.
        return [False] * 5
    sims = est.sims_por_mascara
    best_mask, best_ev = [True] * 5, stand               # plantarse = valor exacto
    comunes = [dice.roll(5) for _ in range(sims)] if est.crn else None
    for mask in MASKS:
        if all(mask):
            continue
        libres = [i for i in range(5) if not mask[i]]
        tot = 0.0
        for s in range(sims):
            fresh = comunes[s] if est.crn else dice.roll(5)
            v = list(valores)
            for i in libres:
                v[i] = fresh[i]
            tot += value(tuple(sorted(v)))
        ev = tot / sims if sims else 0.0
        if ev > best_ev:
            best_ev, best_mask = ev, mask
    return best_mask


def elegir_categoria(dados, disponibles, est: Estrategia, probs) -> Tuple[str, int]:
    if est.seleccion == "voraz":
        return best_category_score(dados, disponibles)
    base = baseline_categorias(tuple(probs))
    cat = max(disponibles, key=lambda c: (score_hand(c, dados) - base[c]))
    return cat, score_hand(cat, dados)


# ----------------------------------------------------------------------------
# Estructuras de juego
# ----------------------------------------------------------------------------
@dataclass
class PlayerState:
    name: str
    scores: Dict[str, Optional[int]] = field(default_factory=lambda: {c: None for c in CATEGORIES_ORDER})
    total: int = 0
    bonus: int = 0
    estrategia: Estrategia = field(default_factory=Estrategia)

    def available_categories(self) -> List[str]:
        return [c for c, v in self.scores.items() if v is None]

    def set_score(self, category: str, points: int) -> None:
        self.scores[category] = points
        upper = sum(v for c, v in self.scores.items() if c in UPPER and v is not None)
        self.bonus = BONUS_POINTS if upper >= BONUS_THRESHOLD else 0           # M4
        self.total = sum(v for v in self.scores.values() if v is not None) + self.bonus


@dataclass
class GameStats:
    total_rolls: int = 0                       # dados REALMENTE lanzados (M2)
    face_hist: Counter = field(default_factory=Counter)
    turnos_yahtzee: int = 0
    turnos: int = 0

    def update_from_roll(self, rolled: List[int]) -> None:
        self.total_rolls += len(rolled)
        self.face_hist.update(rolled)


@dataclass
class GameState:
    players: List[PlayerState]
    dice: Dice = field(default_factory=Dice)
    current_player_idx: int = 0
    dice_values: List[int] = field(default_factory=lambda: [0] * 5)
    locked: List[bool] = field(default_factory=lambda: [False] * 5)
    rolls_left: int = 3
    stats: GameStats = field(default_factory=GameStats)

    def reset_turn(self):
        self.dice_values, self.locked, self.rolls_left = [0] * 5, [False] * 5, 3

    @property
    def current_player(self) -> PlayerState:
        return self.players[self.current_player_idx]

    def advance_player(self):
        self.current_player_idx = (self.current_player_idx + 1) % len(self.players)


def roll_dice(state: GameState) -> None:
    """Lanza los dados no bloqueados y registra solo los lanzados (M2)."""
    idx = [i for i, b in enumerate(state.locked) if not b]
    nuevos = state.dice.roll(len(idx))
    for i, v in zip(idx, nuevos):
        state.dice_values[i] = v
    state.stats.update_from_roll(nuevos)


def play_one_turn(state: GameState, verbose: bool = False) -> None:
    player = state.current_player
    est = player.estrategia
    state.reset_turn()
    roll_dice(state)
    state.rolls_left -= 1
    if verbose:
        print(f"\n{player.name} - Tirada 1:", state.dice_values)
    while state.rolls_left > 0:
        mask = elegir_mascara(state.dice_values, state.rolls_left,
                              player.available_categories(), state.dice, est)
        if all(mask):
            if verbose:
                print(f"{player.name} decide plantarse con:", state.dice_values)
            break
        state.locked = mask[:]
        roll_dice(state)
        state.rolls_left -= 1
        if verbose:
            print(f"{player.name} - Tirada {3 - state.rolls_left}:", state.dice_values, "| keep:", mask)
    cat, sc = elegir_categoria(state.dice_values, player.available_categories(), est, state.dice.probs)
    player.set_score(cat, sc)
    state.stats.turnos += 1
    state.stats.turnos_yahtzee += int(len(set(state.dice_values)) == 1)
    if verbose:
        print(f"-> {player.name} elige '{cat}' y anota {sc}. Total = {player.total}")


def play_game(estrategias: Sequence[Estrategia] = (Estrategia(), Estrategia()),
              probs: Sequence[float] = FAIR, seed: Optional[int] = None,
              verbose: bool = False) -> GameState:
    players = [PlayerState(f"Jugador {i+1}", estrategia=e) for i, e in enumerate(estrategias)]
    state = GameState(players=players, dice=Dice(probs, seed))
    for _ in range(13):
        for _p in players:
            play_one_turn(state, verbose)
            state.advance_player()
    return state


def ganador(state: GameState) -> str:
    a, b = state.players[0].total, state.players[1].total
    return state.players[0].name if a > b else state.players[1].name if b > a else "Empate"


# ----------------------------------------------------------------------------
# M9  Solucion exacta (programacion dinamica) para comparar con Montecarlo
# ----------------------------------------------------------------------------
@lru_cache(maxsize=None)
def dist_lanzamiento(n: int, probs: Tuple[float, ...] = FAIR) -> Dict[Tuple[int, ...], float]:
    """Distribucion exacta (multinomial) del multiconjunto obtenido al lanzar n dados."""
    out: Dict[Tuple[int, ...], float] = {}
    for comb in itertools.combinations_with_replacement(FACES, n):
        c = Counter(comb)
        p = math.factorial(n)
        for k, m in c.items():
            p *= probs[k - 1] ** m / math.factorial(m)
        out[comb] = p
    return out


def valor_optimo(probs: Tuple[float, ...] = FAIR, tiradas_restantes: int = 2,
                 categorias: Sequence[str] = tuple(CATEGORIES_ORDER),
                 valor=None) -> Dict[Tuple[int, ...], float]:
    """V[mano] = valor esperado del mayor puntaje de UNA categoria con la politica
    optima y `tiradas_restantes` relanzamientos (induccion hacia atras)."""
    if valor is None:
        valor = lambda t: max(_score_sorted(c, t) for c in categorias)
    manos = list(itertools.combinations_with_replacement(FACES, 5))
    V = {h: float(valor(h)) for h in manos}
    for _ in range(tiradas_restantes):
        cacheE: Dict[Tuple[int, ...], float] = {}

        def E(kept):
            if kept not in cacheE:
                n = 5 - len(kept)
                cacheE[kept] = sum(p * V[tuple(sorted(kept + o))]
                                   for o, p in dist_lanzamiento(n, probs).items())
            return cacheE[kept]
        V = {h: max(E(tuple(sorted(h[i] for i in range(5) if m[i]))) for m in MASKS) for h in manos}
    return V


def esperanza_turno_optimo(probs=FAIR, categorias=tuple(CATEGORIES_ORDER)) -> float:
    V = valor_optimo(tuple(probs), 2, categorias)
    return sum(p * V[h] for h, p in dist_lanzamiento(5, tuple(probs)).items())


# ----------------------------------------------------------------------------
# CREA  Tres dados iguales: lanzar los otros 2 o detenerse
# ----------------------------------------------------------------------------
def pago_mano(dados, categorias=tuple(CATEGORIES_ORDER)) -> int:
    return max(score_hand(c, dados) for c in categorias)


def tres_iguales_exacto(probs=FAIR, categorias=tuple(CATEGORIES_ORDER)):
    """Para cada cara k: E[pago | relanzar los 2 dados] (exacto, 36 resultados) y
    la tabla de pagos al detenerse con (a, b)."""
    probs = tuple(probs)
    out = {}
    for k in FACES:
        er = sum(probs[x-1] * probs[y-1] * pago_mano([k, k, k, x, y], categorias)
                 for x in FACES for y in FACES)
        stop = {(a, b): pago_mano([k, k, k, a, b], categorias) for a in FACES for b in FACES}
        # probabilidad de cada par (a, b) observado (los otros 2 dados ya lanzados)
        pab = {(a, b): probs[a-1] * probs[b-1] for a in FACES for b in FACES}
        out[k] = dict(relanzar=er, detener=stop, pab=pab)
    return out


def tres_iguales_montecarlo(n: int, probs=FAIR, k: Optional[int] = None, seed: Optional[int] = None,
                            categorias=tuple(CATEGORIES_ORDER)):
    """Estima por Montecarlo (a) E[pago al detenerse], (b) E[pago al relanzar],
    (c) E[pago con decision informada]. Devuelve tambien P(Yahtzee al relanzar)."""
    dice = Dice(probs, seed)
    ex = tres_iguales_exacto(probs, categorias)
    s = r = o = y = 0.0
    for _ in range(n):
        kk = k if k is not None else dice.roll(1)[0]
        a, b = dice.roll(2)
        x, z = dice.roll(2)
        ps, pr = pago_mano([kk, kk, kk, a, b], categorias), pago_mano([kk, kk, kk, x, z], categorias)
        s += ps; r += pr; y += (x == kk and z == kk)
        o += max(ps, ex[kk]["relanzar"])
    return dict(detener=s / n, relanzar=r / n, informada=o / n, p_yahtzee=y / n)


# ----------------------------------------------------------------------------
# M10  P(Yahtzee en un turno): Montecarlo vectorizado y solucion exacta (Markov)
#      Estrategia: conservar la cara mas frecuente y relanzar el resto (2 veces).
# ----------------------------------------------------------------------------
def p_yahtzee_mc(n: int, probs: Sequence[float] = FAIR, seed: Optional[int] = None) -> float:
    import numpy as np
    rng = np.random.default_rng(seed)
    p = np.asarray(probs, dtype=float)
    d = rng.choice(np.arange(1, 7), size=(n, 5), p=p)
    for _ in range(2):
        cnt = np.stack([(d == k).sum(1) for k in range(1, 7)], 1)
        f = 6 - np.argmax(cnt[:, ::-1], 1)                  # cara mas frecuente (empate: mayor)
        libres = d != f[:, None]
        d = np.where(libres, rng.choice(np.arange(1, 7), size=(n, 5), p=p), d)
    return float((d == d[:, [0]]).all(1).mean())


def p_yahtzee_exacto(probs: Sequence[float] = FAIR) -> float:
    """Cadena de Markov de 252 estados (conteos por cara) con 2 relanzamientos."""
    import numpy as np
    probs = tuple(probs)
    estados = [c for c in itertools.product(range(6), repeat=6) if sum(c) == 5]
    idx = {s: i for i, s in enumerate(estados)}

    def multinom(n):
        out = {}
        for c in itertools.product(range(n + 1), repeat=6):
            if sum(c) == n:
                pr = math.factorial(n)
                for k in range(6):
                    pr *= probs[k] ** c[k] / math.factorial(c[k])
                out[c] = pr
        return out

    M = {n: multinom(n) for n in range(6)}
    T = np.zeros((len(estados), len(estados)))
    for s in estados:
        m = max(s)
        f = max(k for k in range(6) if s[k] == m)
        keep = [0] * 6
        keep[f] = m
        for c, pr in M[5 - m].items():
            T[idx[s], idx[tuple(a + b for a, b in zip(keep, c))]] += pr
    v0 = np.array([M[5].get(s, 0.0) for s in estados])
    v = v0 @ np.linalg.matrix_power(T, 2)
    return float(sum(v[i] for i, s in enumerate(estados) if max(s) == 5))
