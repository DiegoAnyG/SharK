# SharK — Revisión científica y refactorización del modelo de factibilidad covalente

## Objetivo

Revisar y refactorizar el modelo matemático de predicción de factibilidad covalente de SharK para que:

1. Distinga claramente entre:
   - reconocimiento reversible,
   - preorganización geométrica,
   - competencia química,
   - formación/estabilidad del aducto.

2. No interprete datos ausentes como evidencia favorable.

3. No presente índices heurísticos como probabilidades físicas.

4. No permita que una afinidad o geometría excelente compense una barrera química incompatible con la reacción.

5. Mantenga compatibilidad con la arquitectura actual siempre que sea posible.

No eliminar las funcionalidades existentes sin necesidad. Priorizar refactorización incremental y mantener los tests existentes, actualizándolos solamente cuando la interpretación científica actual sea incorrecta.

---

# 1. Arquitectura conceptual deseada

El proceso covalente debe representarse conceptualmente como:

    E + I <=> EI -> E-I
              k_inact

Distinguir:

    Ki = koff / kon

del parámetro cinético aparente utilizado en inhibición covalente:

    KI = (koff + kinact) / kon

En el límite de equilibrio rápido:

    koff >> kinact

se puede aproximar:

    KI ~= Ki ~= KD

SharK NO debe inferir directamente KI o KD a partir del score de docking de Vina.

El docking debe utilizarse únicamente como descriptor heurístico de reconocimiento reversible, salvo que exista una calibración experimental explícita.

---

# 2. Nomenclatura propuesta

Actualmente existen varios conceptos llamados "CFI".

Separarlos.

## 2.1 Reactive Geometry Index

Renombrar conceptualmente el índice local actualmente calculado a partir de:

    distance feasibility
    × angular factor
    × activation factor
    × electrophilicity factor

como:

    RGI = Reactive Geometry Index

o un nombre equivalente inequívoco.

Propuesta:

    RGI = f_d * f_theta * f_act * f_elec

Este índice describe compatibilidad local de una geometría de ataque.

NO debe llamarse "Total Covalent Feasibility Index".

---

## 2.2 Pre-reactive Covalent Feasibility Index

Antes de disponer de una barrera de activación válida:

    CFI_pre

Debe incluir únicamente evidencia pre-reactiva:

    CFI_pre = F(S_bind, P_NAC)

Debe reportarse explícitamente como:

    Pre-reactive Covalent Feasibility

Nunca como:

    High Covalent Feasibility
    Covalent probability
    Probability of covalent bond formation

---

## 2.3 Final Covalent Feasibility Index

Solo cuando exista información válida de transición química:

    CFI_final = F(S_bind, P_NAC, S_chem)

Este será el índice global.

No redistribuir automáticamente el peso del término químico cuando éste esté ausente.

Si S_chem no está disponible:

    CFI_final = unavailable / pending

y debe reportarse CFI_pre por separado.

---

# 3. Cambio prioritario: valores faltantes

## Problema actual

Actualmente existen comportamientos equivalentes a:

    if angle is missing:
        f_ang = 0.85

y:

    if electrophilicity is missing:
        f_elec = 1.0

Esto implica matemáticamente:

    missing data -> favorable evidence

lo cual no es aceptable.

## Comportamiento nuevo

Los datos ausentes deben representarse como:

    None
    NA
    not_available

según corresponda a la arquitectura interna.

No asignar por defecto un valor favorable.

### Angular factor

NO:

    angle missing -> f_theta = 0.85

Preferencia:

    angle missing -> f_theta = None

y el RGI completo no debe calcularse como si la geometría angular hubiera sido validada.

Opcionalmente se puede calcular un índice parcial:

    RGI_partial

pero debe etiquetarse claramente como incompleto.

### Electrophilicity factor

NO:

    omega_k missing -> f_elec = 1.0

Preferencia:

    omega_k missing -> f_elec = None

El programa puede reportar:

    Electrophilicity: not evaluated

y continuar con métricas geométricas sin afirmar que la electrofilicidad es favorable.

---

# 4. Afinidad / docking

Actualmente el score de docking se transforma en un score normalizado.

Esto puede mantenerse, pero cambiar la interpretación.

## No afirmar

    docking score = DeltaG_binding
    docking score determines KI
    docking score determines KD

## Utilizar

    S_bind

como descriptor normalizado de reconocimiento reversible.

Una transformación sigmoidal puede mantenerse:

    S_bind =
        1 /
        (1 + exp((dock_score - reference_score) / tau))

pero debe documentarse como una transformación heurística.

No utilizar unidades termodinámicas para el score normalizado final.

No llamarlo probabilidad.

---

# 5. Revisar cálculo numérico de S_bind

Verificar que la implementación corresponda exactamente con la ecuación documentada.

Ejemplo:

    dock_score = -7.244 kcal/mol
    reference = -6.000 kcal/mol
    tau = 1.50 kcal/mol

Entonces:

    x = (-7.244 - (-6.000)) / 1.50
      = -0.82933

    S_bind = 1 / (1 + exp(-0.82933))
           ~= 0.696

Si el software produce aproximadamente 0.81 con esos parámetros, existe una discrepancia entre:

    implementation
    documentation
    parameter values

Agregar test unitario para este caso.

Expected:

    S_bind ~= 0.696

con tolerancia apropiada.

---

# 6. Near-Attack Conformations

Mantener el concepto NAC.

El modelo debe considerar simultáneamente:

    nucleophile-electrophile distance
    attack angle

Preferir una función continua a cortes binarios.

## Propuesta

Para cada frame t:

    NAC_score(t) = f_d(d_t) * f_theta(theta_t)

Luego:

    P_NAC = mean_t[NAC_score(t)]

Esto es preferible a:

    boolean NAC
    +
    penalty based on mean distance

porque evita contabilizar dos veces la distancia y conserva información de la trayectoria.

---

# 7. Distancia NAC

El cutoff de:

    d <= 3.5 Angstrom

puede mantenerse como criterio descriptivo / diagnóstico.

Sin embargo, para scoring usar una función continua.

Ejemplo actual aceptable:

    f_d(d) =
        1 /
        (1 + exp((d - d0) / sigma_d))

con aproximadamente:

    d0 = 3.5 Angstrom

Los parámetros deben poder configurarse.

No presentar 3.5 Angstrom como límite universal de reacción.

---

# 8. Bürgi-Dunitz y geometría de ataque

La función gaussiana actual centrada aproximadamente en 107 grados es conceptualmente preferible al corte booleano 80–145 grados.

Mantener una función del tipo:

    f_theta =
        exp(
            -(theta - theta0)^2 /
            (2 * sigma_theta^2)
        )

con valores por defecto actualmente equivalentes a:

    theta0 = 107 degrees
    sigma_theta = 14 degrees

Pero hacer estos parámetros configurables.

## Importante

No utilizar un único ángulo de Bürgi-Dunitz para todos los mecanismos covalentes.

Futuro diseño deseado:

    attack_geometry = geometry_model(mechanism)

Ejemplos:

    carbonyl_addition
    michael_addition
    SN2
    epoxide_opening
    sulfonyl_substitution
    aromatic_nucleophilic_substitution
    other

Cada mecanismo puede definir:

    theta0
    sigma_theta
    distance model
    reference atoms

---

# 9. Conectividad molecular

Problema actual:

El átomo vecino utilizado para definir el ángulo de ataque puede inferirse por distancia interatómica y prioridad O/N.

Esto no es suficientemente robusto.

## Cambio requerido

Cuando RDKit esté disponible, utilizar la conectividad molecular real:

    target electrophilic atom
        -> bonded neighbors
        -> mechanism-specific reference atom

No seleccionar el átomo adyacente únicamente por:

    1.05 <= distance <= 1.65 Angstrom

ni priorizar automáticamente O/N.

La heurística geométrica puede permanecer como fallback solamente cuando no exista información de conectividad.

El fallback debe generar un warning:

    "Attack angle inferred from geometric connectivity; explicit bond graph unavailable."

---

# 10. Activación del nucleófilo

Problema actual:

La cisteína recibe esencialmente:

    f_act = 1.0

de forma predeterminada.

Esto sobreestima la fracción/reactividad del tiolato.

No asumir:

    CYS == fully activated thiolate

## Modelo propuesto

Separar:

    intrinsic_residue_reactivity

de:

    microenvironment_activation

Ejemplo conceptual:

    f_act =
        f_intrinsic(residue)
        *
        f_environment

El entorno puede considerar, cuando exista información:

    catalytic base proximity
    H-bond geometry
    protonation state
    local electrostatics
    solvent accessibility
    predicted pKa

No es necesario implementar inmediatamente todos estos componentes.

Primera refactorización mínima:

    CYS should NOT default to 1.0

y:

    activation status = uncertain

cuando no exista evidencia suficiente.

---

# 11. "Catalytic dyad"

Actualmente la proximidad de un residuo básico/ácido puede marcar:

    catalytic_dyad_present = True

Esto es demasiado fuerte.

Renombrar preferentemente a:

    candidate_catalytic_base_present

o:

    candidate_activation_partner

La proximidad geométrica por sí sola NO demuestra una díada catalítica.

Para afirmar "catalytic dyad", eventualmente considerar:

    donor-acceptor distance
    H-bond angle
    protonation
    residue identity
    persistence during MD

Hasta entonces utilizar terminología de candidato.

---

# 12. Electrofilicidad

Mantener índices de Fukui/electrofilicidad como información útil de regioselectividad.

Sin embargo:

    missing omega_k != perfect electrophilicity

No usar:

    f_elec = 1.0

cuando omega_k no esté disponible.

## Regioselectividad

Especificar claramente la convención de las cargas/poblaciones.

Si q_k representa población electrónica:

    f_k+ = q_k(N+1) - q_k(N)

Si q_k representa carga atómica:

el signo debe ajustarse de forma consistente.

Documentar qué convención utiliza SharK.

---

# 13. Transition State / Eyring

Utilizar:

    k_chem =
        kappa * (k_B * T / h)
        * exp(-DeltaG_dagger / (R*T))

Si no se utiliza un transmission coefficient calculado:

    kappa = 1

debe declararse explícitamente.

---

# 14. Diferenciar cinética y termodinámica

Eliminar etiquetas que confundan:

    activation barrier

con:

    spontaneity

Especialmente reemplazar:

    "Spontaneous / Rapid Reaction"

cuando la clasificación depende únicamente de DeltaG_dagger.

Una barrera baja informa sobre cinética, no espontaneidad termodinámica.

Usar, por ejemplo:

    Very Rapid Predicted Chemical Step
    Rapid Predicted Chemical Step
    Moderate Predicted Chemical Rate
    Slow Predicted Chemical Step

La favorabilidad termodinámica debe evaluarse por separado utilizando:

    DeltaG_reaction

No inferir espontaneidad a partir de:

    DeltaG_dagger

---

# 15. Umbrales de DeltaG_dagger

Los valores:

    <= 18 kcal/mol
    <= 22 kcal/mol
    <= 25 kcal/mol

pueden mantenerse temporalmente como categorías operacionales.

Pero deben documentarse como:

    operational thresholds

y no como fronteras físicas universales.

Preferentemente hacerlos configurables.

A T = 310.15 K, usar Eyring para convertir la barrera a:

    k_chem
    half-life

y mostrar esas magnitudes junto con la categoría.

La categoría debe ser secundaria respecto al valor físico calculado.

---

# 16. Wiberg Bond Index

No interpretar:

    WBI = 0.97

como:

    97% covalent character

Esto es incorrecto.

Utilizar lenguaje como:

    "WBI = 0.97, consistent with a bond order close to a conventional single bond in the optimized adduct."

El WBI sirve para caracterizar el enlace del aducto optimizado.

No demuestra que la trayectoria hacia dicho aducto sea cinéticamente accesible.

---

# 17. "Orbital overlap"

Si se utiliza una expresión heurística del tipo:

    S_eff =
        S0 *
        cos(theta - theta0) *
        exp(-beta*(d-d0))

NO llamarla:

    orbital overlap integral

porque no se está evaluando explícitamente:

    integral(phi_i* phi_j dV)

Renombrarla a algo como:

    orbital_alignment_score
    geometric_orbital_alignment
    attack_alignment_descriptor

---

# 18. Woodward-Hoffmann

No utilizar:

    S_eff > 0

como prueba universal de:

    Woodward-Hoffmann symmetry allowed

Las reglas de Woodward-Hoffmann no son un criterio universal para todas las reacciones covalentes proteína-ligando.

Eliminar o restringir esa interpretación.

El descriptor puede permanecer como indicador geométrico/orbital heurístico.

---

# 19. Polarización electrónica

No interpretar automáticamente:

    Delta epsilon_LUMO < 0
    Delta mu > 0

como:

    catalytic electric field

Preferir:

    favorable electronic polarization
    environment-induced electronic polarization

Para afirmar catálisis electrostática debería evaluarse explícitamente el campo eléctrico y su proyección sobre la coordenada/dipolo de reacción.

---

# 20. Modelo global del CFI

Evitar un promedio aritmético simple en el que un pilar excelente pueda compensar completamente otro pilar químicamente incompatible.

Modelo recomendado:

    CFI_final =
        S_bind^w_bind
        *
        P_NAC^w_NAC
        *
        S_chem^w_chem

con:

    w_bind + w_NAC + w_chem = 1

Equivalente:

    ln(CFI_final) =
        w_bind * ln(S_bind)
        +
        w_NAC * ln(P_NAC)
        +
        w_chem * ln(S_chem)

Ventaja:

Si:

    S_chem -> 0

entonces:

    CFI_final -> 0

Una excelente afinidad no puede compensar una reacción químicamente imposible.

---

# 21. Tratamiento de ceros

Para evitar:

    log(0)

utilizar un epsilon numérico explícito:

    EPS = 1e-12

y:

    safe_score = max(score, EPS)

No ocultar conceptualmente un cero real.

EPS debe utilizarse únicamente para estabilidad numérica.

---

# 22. Score químico S_chem

Puede derivarse de DeltaG_dagger mediante una transformación monotónica.

Preferible trabajar primero con:

    k_chem

calculado por Eyring.

Después normalizar:

    S_chem = normalization(k_chem)

La normalización debe documentarse y ser configurable.

No tratar S_chem como probabilidad experimental salvo calibración externa.

---

# 23. Relación con k_inact

Cuando sea físicamente razonable y no se esté contabilizando dos veces la misma contribución:

    k_inact_eff ~= P_NAC * k_chem

Interpretar esto como aproximación/modelo, no identidad universal.

Evitar duplicar el efecto de preorganización si el cálculo de DeltaG_dagger ya incluye explícitamente la población conformacional relevante.

Documentar esta limitación.

---

# 24. Parámetro físico final de interés

En inhibición covalente, cuando sea posible obtener/calibrar KI:

    kinact / KI

es un parámetro físicamente interpretable de eficiencia covalente.

SharK puede utilizar CFI como herramienta de screening mientras no disponga de los parámetros necesarios.

No presentar CFI como sustituto experimental de:

    kinact / KI

---

# 25. Output del dossier

Cambiar el dossier para distinguir estados.

Ejemplo cuando NO existe cálculo TS:

    Binding score:
        S_bind = 0.70

    Dynamic reactive preorganization:
        P_NAC = 0.76

    Reactive geometry:
        RGI = 0.82

    Chemical barrier:
        Not evaluated

    CFI_pre:
        0.73

    CFI_final:
        Pending transition-state calculation

    Interpretation:
        Favorable pre-reactive configuration.
        Covalent bond formation has not yet been kinetically validated.

No mostrar:

    "High Covalent Feasibility"

en ausencia de S_chem / DeltaG_dagger.

---

# 26. Output cuando exista TS

Ejemplo:

    S_bind = ...
    P_NAC = ...
    RGI = ...
    DeltaG_dagger = ... kcal/mol
    k_chem = ... s^-1
    t_1/2 = ...
    S_chem = ...
    CFI_final = ...

Reportar además:

    DeltaG_reaction

cuando esté disponible.

Mantener separados:

    kinetic accessibility
    thermodynamic favorability

---

# 27. Eliminar el símbolo %

No mostrar:

    CFI = 77.7%

salvo que exista calibración experimental que permita interpretarlo realmente como probabilidad.

Mostrar:

    CFI = 0.777

o:

    CFI score = 0.777

Las etiquetas cualitativas pueden mantenerse temporalmente si se dejan claras como categorías heurísticas.

---

# 28. Clases / estructuras de datos sugeridas

Considerar estructuras similares a:

    ReactiveGeometryResult:
        distance_score
        angle_score
        activation_score
        electrophilicity_score
        rgi
        completeness
        warnings

    PreReactiveFeasibility:
        binding_score
        p_nac
        cfi_pre
        completeness

    ChemicalFeasibility:
        delta_g_activation
        delta_g_reaction
        temperature
        kappa
        k_chem
        half_life
        chemical_score

    TotalCovalentFeasibility:
        cfi_pre
        cfi_final
        status
        missing_components
        warnings

No es obligatorio utilizar exactamente estos nombres.

Priorizar API limpia y compatibilidad.

---

# 29. Estados de completitud

Introducir estados explícitos, por ejemplo:

    GEOMETRY_ONLY
    PRE_REACTIVE
    QM_PARTIAL
    COMPLETE

o equivalente.

El programa debe saber diferenciar:

    value = low

de:

    value = unknown

No utilizar 0, 0.5 o 1 para representar automáticamente "unknown".

---

# 30. Tests requeridos

Agregar o modificar tests para verificar al menos:

## Test A — Binding normalization

Input:

    dock_score = -7.244
    reference = -6.0
    tau = 1.5

Expected:

    S_bind ~= 0.696

---

## Test B — Missing angle

Input:

    angle = None

Expected:

    angle_score is None

y:

    result is marked incomplete

No aceptar:

    angle_score == 0.85

---

## Test C — Missing electrophilicity

Input:

    omega_k = None

Expected:

    electrophilicity_score is None

No aceptar:

    electrophilicity_score == 1.0

---

## Test D — Missing TS

Input:

    S_bind available
    P_NAC available
    DeltaG_dagger missing

Expected:

    CFI_pre available
    CFI_final is None / unavailable
    status != "High Covalent Feasibility"

---

## Test E — Chemistry dominates impossible reaction

Ejemplo:

    S_bind = 0.95
    P_NAC = 0.95
    S_chem = 0.001

Expected:

    CFI_final remains very low

Una afinidad excelente no debe elevar artificialmente el score global.

---

## Test F — Eyring

Para una barrera definida, verificar numéricamente:

    k_chem =
        k_B*T/h * exp(-DeltaG_dagger/(R*T))

usando:

    T = 310.15 K
    kappa = 1

Agregar tolerancia numérica explícita.

---

## Test G — Kinetic vs thermodynamic labels

Si:

    DeltaG_dagger is low

pero:

    DeltaG_reaction > 0

NO etiquetar automáticamente como:

    spontaneous

---

## Test H — Wiberg wording

Asegurar que el dossier NO contenga patrones como:

    "97% covalent character"

derivados directamente del WBI.

---

# 31. Compatibilidad

No romper:

    CLI
    HTML dossier generation
    existing parsers
    ORCA parsing
    MD analysis
    existing command names

sin necesidad.

Si se cambia una API pública, mantener alias/deprecation temporal cuando sea razonable.

---

# 32. Documentación

Actualizar:

    README.md
    inline docstrings
    HTML dossier explanatory text
    mathematical description

para que la implementación y la documentación utilicen las mismas ecuaciones.

No permitir discrepancias entre:

    equation shown to user
    code implementation
    tests

---

# 33. Filosofía científica del modelo

SharK debe tratar los tres componentes principales como ortogonales:

    Recognition
    +
    Reactive preorganization
    +
    Chemical accessibility

La lógica científica deseada es:

    Does it bind in a plausible pose?

    Does the solvated complex persistently reach a reactive geometry?

    Is the correct atom electronically susceptible to attack?

    Is the nucleophile plausibly activated?

    Is the chemical barrier kinetically accessible?

    Is the resulting adduct electronically consistent with bond formation?

No inferir covalencia únicamente del docking.

No inferir reacción únicamente de proximidad.

No inferir espontaneidad únicamente de DeltaG_dagger.

No inferir percent covalent character directamente de WBI.

---

# 34. Prioridades de implementación

Implementar en este orden:

### Priority 1
Eliminar defaults optimistas para datos faltantes:

    f_ang = 0.85
    f_elec = 1.0

cuando dichos datos realmente son desconocidos.

### Priority 2
Separar:

    RGI
    CFI_pre
    CFI_final

### Priority 3
No redistribuir pesos del TS cuando el TS está ausente.

### Priority 4
Cambiar el modelo global de suma ponderada a media geométrica ponderada.

### Priority 5
Corregir terminología:

    spontaneous
    catalytic dyad
    orbital overlap integral
    WBI percent covalent character
    catalytic electric field

### Priority 6
Refactorizar activación del nucleófilo, especialmente:

    CYS != automatically fully activated

### Priority 7
Usar conectividad real de RDKit para definir la geometría de ataque.

### Priority 8
Preparar arquitectura mechanism-specific para distintos warheads.

---

# 35. Restricción importante

No convertir esta revisión en una reescritura completa del proyecto.

Primero:

    inspect current implementation
    identify affected functions
    make minimal coherent changes
    update tests
    run full test suite
    report regressions

Antes de modificar código, generar un breve plan indicando:

    files to modify
    functions/classes affected
    API changes
    tests to update/add

Luego implementar.

---

# 36. Resultado esperado del agente

Al terminar, entregar:

1. Resumen de los cambios realizados.
2. Archivos modificados.
3. Ecuaciones finales realmente implementadas.
4. Tests nuevos/modificados.
5. Resultado del test suite.
6. Cualquier incompatibilidad introducida.
7. Aspectos científicos que continúan siendo aproximaciones heurísticas.
8. TODOs recomendados para una futura versión.

