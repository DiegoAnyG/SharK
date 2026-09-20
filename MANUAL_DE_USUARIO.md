# Manual de Usuario Exhaustivo de SharK
**Screening & Holistic Analysis of Reactivity and Kinetics (SharK)**  
*Plataforma Integral para el Descubrimiento y Caracterización de Inhibidores Covalentes Dirigidos (TCIs)*

---

## 1. Introducción y Arquitectura de Tiers

SharK es una plataforma computacional modular diseñada para desacoplar y cuantificar de manera rigurosa los fenómenos que rigen la inhibición covalente dirigida: el reconocimiento reversible inicial, la estabilidad conformacional reactiva y la cinética de formación del enlace químico mediada por mecánica cuántica.

El flujo de trabajo se organiza en cuatro niveles jerárquicos (*Tiers*):

```
+-----------------------------------------------------------------------------+
| TIER 1: Reconocimiento Reversible y Acoplamiento Molecular                 |
|   - Acoplamiento molecular con AutoDock Vina / Smina                        |
|   - Puntuacion de union Delta G_dock y afinidad sigmoidal S_bind            |
|   - Equilibrio tautomerico de Boltzmann en disolucion acuosa                |
+-----------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------+
| TIER 2: Dinamica Molecular y Preorganizacion Reactiva                      |
|   - Dinamica molecular en disolucion explicita con GROMACS                  |
|   - Probabilidad de Conformacion Casi de Ataque (P_NAC)                     |
|   - Agrupamiento conformacional Daura/GROMOS (RMSD <= 1.5 A)                |
|   - Extraccion del medoide dominante y del medoide reactivo ponderado       |
+-----------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------+
| TIER 3: Reactividad Quimica Conceptual (CDFT) y Cluster de Bolsillo         |
|   - Calculos DFT de orbitales frontera HOMO / LUMO en ligando y cluster     |
|   - Dureza quimica (eta), suavidad (S) e indice de electrofilia global (w)  |
|   - Funciones de Fukui condensadas (f_k+) y suavidad local (s_k+)           |
|   - Evaluacion de simetria de fase orbital y polarizacion de bolsillo       |
+-----------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------+
| TIER 4: Estado de Transicion (TS), Kinetica de Eyring y Aducto Covalente    |
|   - Preflight de coordenada de reaccion y prevencion de pares incompatibles |
|   - Barrido de superficie de energia potencial (PES scan relajado) en ORCA  |
|   - Optimizacion de punto de silla (! OptTS Freq) y verificacion vibracional|
|   - Barrera de activacion Delta G‡, constante k_chem y vida media t_1/2     |
|   - Optimizacion del aducto covalente y analisis de orden de enlace Wiberg   |
+-----------------------------------------------------------------------------+
```

---

## 2. Glosario Exhaustivo de Metricas y Fundamentos Quimicos

A continuacion se detallan todas las variables, su definicion matematica, su significado fisicoquimico y como se interrelacionan para determinar la viabilidad del compuesto.

### 2.1 Metricas de Reconocimiento Reversible (Tier 1)

* **$\Delta G_{\text{dock}}$ (Energia Libre de Acoplamiento, $\text{kcal/mol}$):**
  Estimacion de la afinidad no covalente entre el ligando y la diana en su estado nativo antes de la reaccion. Valores tipicos para inhibidores potentes oscilan entre $-7.0$ y $-11.0\text{ kcal/mol}$.
* **$S_{\text{bind}}$ (Puntuacion Normalizada de Reconocimiento, adimensional $[0, 1]$):**
  Transformacion sigmoidal de Fermi-Dirac de la energia de docking:
  $$S_{\text{bind}} = \frac{1}{1 + \exp\left( \frac{\Delta G_{\text{dock}} - \Delta G_{\text{ref}}}{\tau} \right)}$$
  donde $\Delta G_{\text{ref}} = -6.0\text{ kcal/mol}$ y $\tau = 1.5\text{ kcal/mol}$. Permite comparar ligandos en una escala normalizada donde valores $> 0.70$ indican un reconocimiento favorable.

### 2.2 Metricas de Preorganizacion Dinamica (Tier 2)

* **Distancia de Ataque ($d_{\text{att}}$, $\text{\AA}$):**
  Distancia euclidea tridimensional entre el atomo nucleofilico del residuo diana (ejemplo: `THR309:OG1`, `CYS145:SG`) y el centro electrofilico del ligando (ejemplo: carbono carbonilico, nitrogeno o carbono del warhead).
* **Angulo de Bürgi-Dunitz ($\theta_{\text{BD}}$, grados):**
  Angulo de aproximacion del nucleofilo respecto al plano y enlace del centro electrofilico. Para adiciones nucleofilicas a grupos carbonilo o conjugados, la trayectoria optima se situa en $105^\circ \pm 15^\circ$.
* **Conformacion Casi de Ataque (Near-Attack Conformation, NAC):**
  Estado conformacional prerreactivo donde el ligando se ubica dentro de la distancia de reaccion y con la orientacion angular requerida para la formacion del enlace:
  $$d_{\text{Nu-E}} \le 3.50\text{ \AA} \quad \text{y} \quad \theta_{\text{BD}} \in [80^\circ, 130^\circ]$$
* **$P_{\text{NAC}}$ (Probabilidad de NAC en Dinamica Molecular, porcentaje $\%$):**
  Fraccion de fotogramas (*frames*) de la trayectoria de dinamica molecular solvatada en la que se cumplen simultaneamente los criterios de distancia y angulo.
  * $P_{\text{NAC}} \ge 10.0\%$: Excelente preorganizacion; la diana cataliza o asiste entropicamente el encuentro reactivo.
  * $1.0\% \le P_{\text{NAC}} < 10.0\%$: Preorganizacion moderada; poblacion reactiva accesible transitoriamente.
  * $P_{\text{NAC}} < 1.0\%$: Pobre orientacion en disolucion; el ligando pasa la mayor parte del tiempo en conformaciones no productivas.
* **Medoide Dominante vs. Medoide Reactivo:**
  * *Medoide Dominante:* Centroide del conglomerado (*cluster*) mas poblado de toda la trayectoria. Representa el estado termodinamico de mayor abundancia en disolucion.
  * *Medoide Reactivo:* Estructura representativa extraida exclusivamente del subconjunto de fotogramas que cumplen criterios NAC. SharK utiliza este medoide reactivo como punto de partida prioritario para el cluster cuantico (Tier 4), garantizando que el escaneo hacia el estado de transicion comience desde una geometria orientada a la reaccion.

### 2.3 Metricas de Reactividad Cuantica Conceptual (CDFT) (Tier 3)

* **$E_{\text{HOMO}}$ y $E_{\text{LUMO}}$ ($\text{eV}$):**
  Energias de los orbitales moleculares ocupado mas alto (HOMO) y desocupado mas bajo (LUMO).
* **Brecha HOMO-LUMO ($\Delta E_{\text{gap}} = E_{\text{LUMO}} - E_{\text{HOMO}}$, $\text{eV}$):**
  Indicador de estabilidad molecular. Brechas intermedias ($2.5$ a $4.5\text{ eV}$) reflejan un equilibrio ideal entre reactividad covalente y estabilidad metabolica en disolucion.
* **Dureza Quimica ($\eta$) y Suavidad ($S$):**
  $$\eta \approx \frac{E_{\text{LUMO}} - E_{\text{HOMO}}}{2}, \quad S = \frac{1}{2\eta}$$
  Miden la resistencia de la densidad electronica a ser deformada. Nucleofilos blandos (como tioles de cisteina) reaccionan preferentemente con electrofilos blandos, mientras que alcoholes (treonina, serina) requieren acoplamiento adecuado de dureza.
* **Indice de Electrofilia Global ($\omega$, $\text{eV}$):**
  $$\omega = \frac{\mu^2}{2\eta}, \quad \text{donde } \mu = \frac{E_{\text{HOMO}} + E_{\text{LUMO}}}{2}$$
  Cuantifica la propension intrinseca del ligando a adquirir densidad electronica. Valores de $\omega > 1.5\text{ eV}$ clasifican al compuesto como un electrofilo moderado a fuerte.
* **Funcion de Fukui Condensada ($f_k^+$) y Suavidad Local ($s_k^+$):**
  Indica que atomo especifico de la molecula es el mas susceptible de sufrir un ataque nucleofilico. Un valor maximo de $f_k^+$ en el atomo previsto confirma la regiospecificidad de la reaccion.

### 2.4 Metricas de Estado de Transicion y Kinetica Cuantica (Tier 4)

* **Barrera de Activacion ($\Delta G^\ddagger$ o $\Delta E^\ddagger$, $\text{kcal/mol}$):**
  Diferencia de energia entre el estado prerreactivo y el punto de silla de primer orden (estado de transicion).
  * $\Delta G^\ddagger < 20.0\text{ kcal/mol}$: Reaccion muy rapida a temperatura ambiente ($298\text{ K}$).
  * $20.0 \le \Delta G^\ddagger \le 25.0\text{ kcal/mol}$: Kinetica optima para farmacos covalentes con inactivacion selectiva y baja reactividad no especifica.
  * $\Delta G^\ddagger > 28.0\text{ kcal/mol}$: Reaccion termodinamicamente impedida o excesivamente lenta.
* **Constante de Velocidad Kinetica ($k_{\text{chem}}$, $\text{s}^{-1}$):**
  Calculada mediante la teoria del estado de transicion de Eyring:
  $$k_{\text{chem}} = \frac{k_B T}{h} \exp\left( - \frac{\Delta G^\ddagger}{R T} \right)$$
* **Vida Media de Inactivacion Cuantica ($t_{1/2}$):**
  Tiempo requerido para inactivar el $50\%$ de la poblacion diana a saturacion:
  $$t_{1/2} = \frac{\ln 2}{k_{\text{chem}}}$$
* **Energia de Reaccion ($\Delta G_{\text{rxn}}$ o $\Delta E_{\text{rxn}}$, $\text{kcal/mol}$):**
  Diferencia termodinamica entre el aducto covalente final y los reactivos iniciales. Valores negativos ($\Delta G_{\text{rxn}} < 0$) confirman que la formacion del enlace es exergonica.
* **Verificacion del Punto de Silla (Modos Vibracionales):**
  Un estado de transicion valido debe poseer estricta y unicamente **una frecuencia vibracional imaginaria significativa** ($\nu_i < -15.0\text{ cm}^{-1}$), cuyo vector de transicion proyecte sobre la coordenada de formacion del enlace nucleofilo-electrofilo.

### 2.5 Metricas del Aducto Covalente (Tarjeta 4: "Formed Bond Nature")

* **Orden de Enlace de Wiberg (WBO, adimensional):**
  Medida ab initio o derivada del solapamiento electronico entre el nucleofilo y el electrofilo en la geometria de producto optimizada.
  * $\text{WBO} \approx 0.85 - 1.05$: Enlace covalente sigma sencillo clasico.
  * $\text{WBO} < 0.30$: Contacto debil o no covalente (no enlazado).
* **Transferencia de Carga Neta ($\Delta q$, electrones $e$):**
  Flujo de densidad electronica entre el ligando y la proteina al formarse el aducto.
* **Porcentaje de Covalencia ($\%$):**
  Grado de comparticion electronica normalizado basado en el orden de enlace y la diferencia de electronegatividades.

### 2.6 Indices de Factibilidad Covalente Global (CFI)

* **$CFI_{\text{pre}}$ (Indice Previo al Estado de Transicion, $[0, 1]$):**
  Evalua la viabilidad combinando docking, dinamica molecular y CDFT:
  $$CFI_{\text{pre}} = w_1 S_{\text{bind}} + w_2 P_{\text{NAC}} + w_3 \text{CDFT\_score}$$
* **$CFI_{\text{final}}$ (Indice de Factibilidad Covalente Integral, $[0, 1]$):**
  Sintesis holistica final que penaliza o promueve el candidato en funcion de la barrera de activacion cuantica real:
  * Si $\Delta G^\ddagger \le 22.0\text{ kcal/mol}$, consolida una alta puntuacion.
  * Si la barrera es prohibitiva ($\Delta G^\ddagger > 30.0\text{ kcal/mol}$), el $CFI_{\text{final}}$ decae severamente, independientemente de lo favorable que fuera el docking.

---

## 3. Modos de Ejecucion Practicos

SharK puede ejecutarse en diferentes modalidades segun el nivel de detalle computacional requerido.

### Modo 1: Cribado Rapido Basado en Docking (`--fast-analysis`)
Para evaluar rapidamente decenas o cientos de compuestos a partir de poses de docking sin ejecutar dinamica molecular:

```bash
shark \
  --session /ruta/a/sesion_docking.tar.gz \
  --fast-analysis \
  --target-residue THR309 \
  --work-dir ./shark_fast_results
```

### Modo 2: Estandar de Oro Simple con Dinamica Molecular Externa (`--simple-gold-standard`)
Cuando ya se dispone de una trayectoria calculada en GROMACS (`.xtc` y `.gro`):

```bash
shark \
  --session /ruta/a/sesion_docking.tar.gz \
  --topology /ruta/a/sistema_solvatado.gro \
  --trajectory /ruta/a/trayectoria_produccion.xtc \
  --simple-gold-standard \
  --target-residue THR309 \
  --work-dir ./shark_md_results
```

*Parametros de clustering utiles:*
* `--cluster-cutoff 1.5`: Radio de corte RMSD en Angstroms para el algoritmo Daura/GROMOS (por defecto: 1.5 A).
* `--cluster-stride 5`: Muestreo de fotogramas para optimizar velocidad sin perder resolucion.

### Modo 3: Flujo Completo con Preparacion y Ejecucion de MD (`--full-gold-standard`)
Para preparar el sistema desde el PDB cristalino crudo, parametrizar el ligando con GAFF/OpenFF, simular en GROMACS y analizar:

```bash
shark \
  --session /ruta/a/sesion_docking.tar.gz \
  --full-gold-standard \
  --time-ns 20.0 \
  --target-residue THR309 \
  --work-dir ./shark_full_results
```

### Modo 4: Modelado Cuantico de Estado de Transicion y Kinetica (`--tier-4-ts --execute`)
Para ejecutar el barrido de coordenada de reaccion (relaxed scan) y la optimizacion del estado de transicion con ORCA:

```bash
shark \
  --session /ruta/a/sesion_docking.tar.gz \
  --topology /ruta/a/sistema.gro \
  --trajectory /ruta/a/trayectoria.xtc \
  --simple-gold-standard \
  --target-residue THR309 \
  --tier-4-ts \
  --qm-model minimal \
  --theory r2SCAN-3c \
  --solvent Water \
  --nprocs 4 \
  --execute \
  --work-dir ./shark_ts_results
```

---

## 4. Estrategias de Reanudacion y Continuacion

Un escenario comun es disponer de resultados parciales (por ejemplo, el docking y la dinamica molecular ya completados) y desear agregar el analisis cuantico de Tier 4 o completar un calculo interrumpido sin repetir etapas previas.

### Caso A: Como Obtener los Resultados de la Tarjeta 4 ("Formed Bond Nature")

#### Por que aparece "Status: NOT CALCULATED"?
En cumplimiento riguroso de los estandares de auditoria cientifica (`INV-007` e `INV-010`), SharK **rechaza fabricar o interpolar artificialmente ordenes de enlace covalente de Wiberg, transferencias de carga o naturalezas de enlace si solo se dispone de un contacto prerreactivo** ($d > 2.0\text{ \AA}$).

La Tarjeta 4 se activa exclusivamente cuando se cumple alguna de las siguientes condiciones:
1. El escaneo coordinado (relaxed scan) concluyo exitosamente hasta la distancia de enlace formado ($d \le 1.80\text{ \AA}$, tipicamente el paso 18 a $1.45\text{ \AA}$).
2. O bien, se dispone de una optimizacion convergida del aducto covalente (`03_adduct_opt.out`).

Si el scan se interrumpio antes de llegar al paso final (por ejemplo, en el paso 7 a $d = 2.74\text{ \AA}$), SharK clasifica honestamente el estado como "Pre-reactive Contact (Unreacted)" y reporta la tarjeta como no calculada.

#### Procedimiento Paso a Paso para Calcular la Tarjeta 4:

**Opcion 1: Usando la nueva bandera `--optimize-adduct`:**
Ejecute SharK solicitando la optimizacion directa del producto aducto:

```bash
shark \
  --session /ruta/a/sesion.tar.gz \
  --topology /ruta/a/sistema.gro \
  --trajectory /ruta/a/trayectoria.xtc \
  --simple-gold-standard \
  --target-residue THR309 \
  --tier-4-ts \
  --optimize-adduct \
  --execute \
  --work-dir /ruta/a/su_trabajo
```

**Opcion 2: Optimizando el aducto manualmente con ORCA:**
1. Ingrese al subdirectorio `transition_state/` de su trabajo:
   ```bash
   cd /ruta/a/su_trabajo/transition_state
   ```
2. Si existe el archivo `03_adduct_opt_template.inp`, construya `03_adduct_opt.inp` copiando la plantilla o utilizando las coordenadas del ultimo punto disponible con la distancia contraida a $1.45\text{ \AA}$.
3. Ejecute ORCA sobre la entrada del aducto:
   ```bash
   orca 03_adduct_opt.inp > 03_adduct_opt.out
   ```
4. Una vez terminado con `ORCA TERMINATED NORMALLY`, re-ejecute SharK sin calcular MD para regenerar el informe:
   ```bash
   shark \
     --session /ruta/a/sesion.tar.gz \
     --topology /ruta/a/sistema.gro \
     --trajectory /ruta/a/trayectoria.xtc \
     --simple-gold-standard \
     --target-residue THR309 \
     --tier-4-ts \
     --work-dir /ruta/a/su_trabajo
   ```
   SharK detectara inmediatamente `03_adduct_opt.out`, calculara el orden de enlace y la transferencia de carga, y la Tarjeta 4 quedara calculada al $100\%$.

### Caso B: Reanudar un Calculo de ORCA Interrumpido (OptTS o Scan)

Si una optimizacion de OptTS alcanzo el limite de ciclos o se interrumpio, ORCA genera un archivo de reinicio con la geometria actual y el Hessiano cartesian (`02_optts.opt` o `02_optts.carthess`).

SharK genera automaticamente un script de reanudacion:
```bash
cd /ruta/a/su_trabajo/transition_state
./resume_ts.sh
```

El script prepara `02_optts_resume.inp` reutilizando las coordenadas mas recientes y el operador `%geom InHess read` para no perder el calculo del Hessiano previo.

### Caso C: Regenerar el Dossier HTML a partir de `evidence.json`

Si ya dispone de todos los datos en `evidence.json` y solo desea actualizar el informe visual (por ejemplo, despues de una actualizacion de estilo o visualizador 3D):

```bash
shark --report-from-evidence /ruta/a/su_trabajo/evidence.json --html /ruta/a/su_trabajo/dossier.html
```
Este comando se ejecuta en menos de 2 segundos sin reejecutar ninguna simulacion ni calculo mecanocuantico.

---

## 5. Monitoreo en Tiempo Real y Control de Procesos Parasitos

### 5.1 Barra de Progreso y Tiempo Restante (ETA)

Para evitar la incertidumbre sobre si un calculo de ORCA se encuentra en ejecucion o congelado, SharK monitorea continuamente la salida estandar de ORCA en tiempo real:

* **Durante el Escaneo Relajado (Scan):**
  Muestra el porcentaje completado, el paso actual, la distancia interatomica instantanea y el tiempo estimado para finalizar:
  ```text
  [ORCA SCAN] [████████░░░░░░░░] 44% (Step 8/18, d=2.63 Å) | Elapsed: 2m 15s | ETA: 2m 48s
  ```
* **Durante la Optimizacion de Geometria o OptTS:**
  Muestra el ciclo de optimizacion activo y la energia electronica instantanea:
  ```text
  [ORCA OPT] Cycle 12 | E: -1042.1274 Eh | Elapsed: 4m 30s
  ```
* **Durante el Calculo del Hessiano y Frecuencias:**
  ```text
  [ORCA FREQ] Calculating Hessian & Frequencies | Elapsed: 5m 12s
  ```
* **Durante las Iteraciones SCF:**
  ```text
  [ORCA SCF] Iteration 6 | Elapsed: 45s
  ```

### 5.2 Prevencion y Eliminacion de Procesos Huérfanos / Parasitos

Cuando un calculo de ORCA en paralelo (OpenMPI) es abortado con `Ctrl+C` o sufre un fallo de lectura, los subprocesos MPI (`mpirun`, `orca_scf_mpi`, `orca_prop_mpi`, etc.) pueden quedar corriendo en segundo plano consumiendo memoria RAM y nucleos de CPU.

SharK implementa dos mecanismos robustos de defensa:

1. **Aislamiento en Grupo de Procesos (`process-group kill`):**
   Todos los subprocesos se lanzan en un grupo de procesos independiente (`os.setpgid`). Si el proceso principal se detiene, se emite una senal `SIGTERM` seguida de `SIGKILL` a todo el grupo, garantizando que ningun hilo de OpenMPI quede con vida.
2. **Herramienta Integrada de Limpieza (`--kill-orphans`):**
   Si sospecha que quedaron procesos parasitos de ejecuciones previas o sesiones bloqueadas, ejecute simplemente:
   ```bash
   shark --kill-orphans
   ```
   SharK escaneara la tabla de procesos del sistema operativo, identificara cualquier binario perteneciente a ORCA o a sus trabajadores MPI derivados y los terminara de forma segura:
   ```text
   [CLEANUP] Terminated 8 orphaned ORCA/MPI processes.
   ```

---

## 6. Seguridad Quimica y de Recursos (Especificaciones del Auditor)

SharK incorpora las directivas de fiabilidad de `SHARK_TIER4_ORCA_RELIABILITY_SPEC.md`:

### 6.1 Preflight de Coordenada de Reaccion
Antes de generar archivos de entrada para ORCA, SharK valida que el par nucleofilo-electrofilo sea quimicamente congruente con el mecanismo declarado:
* **Mecanismos soportados:**
  * `carbonyl_addition`: Ataque de Nu (O, S, N) sobre carbono electrofilico (C).
  * `michael_addition`: Adicion 1,4 conjugada sobre carbono beta (C).
  * `furoxan_heterocycle_attack`: Ataque a anillo de benzofuroxano (N o C).
  * `aromatic_substitution_snar`: Sustitucion nucleofilica aromatica.
  * `generic_covalent_addition`: Adicion covalente generica.
* **Prevencion de Enlaces Accidentales:**
  Si para un mecanismo de adicion a carbonilo el algoritmo geométrico seleccionase accidentalmente un oxigeno del ligando como centro diana (coordenada O-O), SharK aborta de inmediato con estado **`BLOCKING`**, impidiendo que ORCA gaste tiempo de computo en una reaccion imposible.

Para especificar de manera inequivoca el atomo electrofilico del ligando:
```bash
shark ... --electrophile-atom C7 --mechanism carbonyl_addition
```

### 6.2 Gestion Inteligente de Memoria RAM (`%maxcore`)
Para evitar fallos catastroficos por falta de memoria (*Out of Memory / OOM Killer*), SharK calcula dinamicamente la memoria por proceso basandose en la **memoria RAM disponible real del sistema** en el momento del lanzamiento, aplicando una fraccion de seguridad del $65\%$:
$$\text{maxcore\_mb} = \frac{\text{RAM\_disponible} \times 0.65}{\text{nprocs}}$$

Si la memoria disponible por hilo resultase inferior a $350\text{ MB}$, SharK reduce automaticamente el numero efectivo de hilos en lugar de asfixiar los procesos de ORCA.

### 6.3 Frecuencia de Recalculo del Hessiano (`--recalc-hess`)
Durante la busqueda del estado de transicion (`! OptTS`), el recalculado del Hessiano exacto puede ajustarse:
```bash
shark ... --recalc-hess 25   # Por defecto: recalcula cada 25 ciclos (optimo en velocidad)
shark ... --recalc-hess 10   # Mayor rigor para sistemas con modos acoplados complejos
shark ... --recalc-hess 5    # Maxima precision en convergencias dificiles
```

### 6.4 Huella Digital de Calculo (*Fingerprint-Aware Cache*)
SharK calcula una huella digital determinista SHA-256 de las coordenadas, atomos reactivos, multiplicidad, carga, nivel de teoria y solvente (`01_scan.meta.json` y `02_optts.meta.json`).
Si el usuario modifica el atomo electrofilico o la geometria del ligando, SharK invalida automaticamente los resultados previos y reejecuta el calculo limpio, evitando mezclar resultados de quimicas incompatibles.

---

## 7. Preguntas Frecuentes y Solucion de Problemas

### Pregunta 1: "ORCA se detuvo con el error: `ERROR (SHARK): Failed to read input file (01_scan.SHARKINP.tmp)`"
* **Causa:** Colision de lectura concurrente de archivos temporales en `orca_prop_mpi` cuando se utiliza un numero elevado de procesos MPI en un mismo directorio sobre sistemas de archivos con latencia.
* **Solucion:** Reduzca el numero de hilos a 4 (`--nprocs 4`). SharK ejecutara el calculo con menor contencion de I/O y completara el escaneo sin errores de lectura.

### Pregunta 2: "El residuo diana (ej. THR309) o el ligando se ven extraños en el visor 3D"
* El visor molecular interactivo 3D del informe HTML utiliza el codigo de colores elemental universal CPK (Carbono: pizarra oscuro `#475569`, Oxigeno: rojo `#ef4444`, Nitrogeno: azul `#3b82f6`, Azufre: amarillo `#eab308`). El nucleofilo reactivo diana (ej. `OG1`) se resalta con un marcador cubico distintivo y un enlace guia directo hacia el electrofilo para permitir la inspeccion visual inmediata de la geometria de ataque.

### Pregunta 3: "Deseo pausar el analisis y continuar mañana sin perder lo calculado"
* Todos los artefactos intermedios se almacenan de forma persistente en el directorio de trabajo (`--work-dir`). Al volver a ejecutar el comando de SharK apuntando al mismo directorio, el sistema detecta los calculos finalizados validos (mediante su huella SHA-256) y continua inmediatamente en la etapa pendiente.

---
*Manual de Usuario de SharK — Diseñado para computo cientifico robusto, reproducible y autonomo.*

