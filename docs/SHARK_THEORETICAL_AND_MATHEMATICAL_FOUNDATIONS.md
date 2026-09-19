# Fundamentos Teóricos, Físico-Matemáticos y Químicos de SharK

Este documento constituye el manual de referencia formal sobre los principios de mecánica cuántica, dinámica molecular clásica, termodinámica estadística y teoría de reactividad química implementados en la plataforma **SharK (Screening & Holistic Analysis of Reactivity and Kinetics)** para el descubrimiento y optimización de inhibidores covalentes dirigidos (*Targeted Covalent Inhibitors*, TCIs).

---

## 1. El Paradigma de los Tres Pilares del Diseño Covalente

El mecanismo de inhibición covalente dirigida se rige por un proceso bi-etápico formal de aproximación no covalente y subsiguiente formación del enlace químico:

$$\mathrm{E} + \mathrm{I} \underset{k_{\text{off}}}{\overset{k_{\text{on}}}{\rightleftharpoons}} \mathrm{E}\cdot\mathrm{I} \xrightarrow{k_{\text{chem}}} \mathrm{E}-\mathrm{I}$$

donde:
- $\mathrm{E}$ representa la macromolécula diana (enzima/receptor).
- $\mathrm{I}$ es el inhibidor portador de un grupo electrofílico reactivo (*warhead*).
- $\mathrm{E}\cdot\mathrm{I}$ es el complejo reversible prerreactivo en disolución.
- $\mathrm{E}-\mathrm{I}$ es el aducto covalente final modificado.
- La constante de disociación en equilibrio del paso reversible es $K_i = k_{\text{off}} / k_{\text{on}}$.
- La velocidad de inactivación covalente a saturación es $k_{\text{inact}} \approx k_{\text{chem}}$.
- La eficiencia catalítica de inactivación está gobernada por la razón de segundo orden:

$$\frac{k_{\text{inact}}}{K_i} \approx \frac{k_{\text{chem}}}{K_i}$$

SharK desacopla cuantitativamente este fenómeno en **Tres Pilares Ortogonales**, evaluados de manera secuencial y unificados en el **Índice de Factibilidad Covalente (CFI)**:

```
[Pilar 1: Reconocimiento Reversible]  --> Afinidad no covalente inicial S_bind
                 +
[Pilar 2: Preorganización Dinámica]   --> Probabilidad de Near-Attack Conformation P_NAC en MD
                 +
[Pilar 3: Cinética Química TS]       --> Barrera cuántica Delta G‡ y velocidad k_chem en ORCA
                 ||
                 \/
[Índice de Factibilidad Covalente CFI] --> Factibilidad global ponderada (CFI_pre y CFI_final)
```

---

## 2. Pilar 1: Reconocimiento Reversible y Termodinámica de Acoplamiento

### 2.1 Función de Puntuación Empírica de Acoplamiento Molecular

El complejo inicial se modela mediante acoplamiento molecular (*molecular docking*) utilizando funciones de energía libre semiempírica (AutoDock Vina / Smina). La energía libre estimada de unión se descompone en una suma de potenciales de interacción por pares entre átomos del receptor ($i$) y del ligando ($j$):

$$\Delta G_{\text{dock}} = \sum_{i \in \text{rec}} \sum_{j \in \text{lig}} f_{\text{inter}}(r_{ij}) + w_{\text{tor}} N_{\text{rot}}$$

donde $r_{ij} = \|\mathbf{r}_i - \mathbf{r}_j\|$, $N_{\text{rot}}$ es el número de enlaces rotables activos del ligando, y $f_{\text{inter}}$ combina cinco funciones por tramos ajustadas empíricamente:

1. **Término gaussiano de atracción de corto alcance:**
   $$f_{\text{gauss}_1}(r_{ij}) = w_1 \exp\left( - \left(\frac{r_{ij} - (R_i + R_j)}{0.5\text{ \AA}}\right)^2 \right)$$

2. **Término gaussiano de atracción de medio alcance:**
   $$f_{\text{gauss}_2}(r_{ij}) = w_2 \exp\left( - \left(\frac{r_{ij} - (R_i + R_j) - 3.0\text{ \AA}}{2.0\text{ \AA}}\right)^2 \right)$$

3. **Repulsión estérica (Pauli / solapamiento nuclear):**
   $$f_{\text{rep}}(r_{ij}) = \begin{cases} w_{\text{rep}} (r_{ij} - (R_i + R_j))^2 & \text{si } r_{ij} < R_i + R_j \\ 0 & \text{si } r_{ij} \ge R_i + R_j \end{cases}$$

4. **Interacción hidrofóbica:**
   $$f_{\text{hydro}}(r_{ij}) = \begin{cases} w_{\text{hydro}} & \text{si } d_{ij} < 0.5\text{ \AA} \\ w_{\text{hydro}} \left(1 - \frac{d_{ij} - 0.5}{1.0}\right) & \text{si } 0.5\text{ \AA} \le d_{ij} \le 1.5\text{ \AA} \\ 0 & \text{si } d_{ij} > 1.5\text{ \AA} \end{cases}$$
   donde $d_{ij} = r_{ij} - (R_i + R_j)$ para átomos apolares (C no enlazado a heteroátomos).

5. **Enlaces de hidrógeno:**
   $$f_{\text{hbond}}(r_{ij}) = \begin{cases} w_{\text{hbond}} & \text{si } d_{ij} < -0.7\text{ \AA} \\ w_{\text{hbond}} \left(-\frac{d_{ij}}{0.7}\right) & \text{si } -0.7\text{ \AA} \le d_{ij} \le 0\text{ \AA} \\ 0 & \text{si } d_{ij} > 0\text{ \AA} \end{cases}$$
   entre pares donador-aceptor compatibles.

### 2.2 Normalización Sigmoidal de Reconocimiento ($S_{\text{bind}}$)

Para transformar la energía libre empírica $\Delta G_{\text{dock}}$ (con unidades en $\text{kcal/mol}$) en una variable de factibilidad probabilística adimensional en el intervalo $(0, 1]$, SharK emplea una función de Fermi-Dirac normalizada:

$$S_{\text{bind}} = \frac{1}{1 + \exp\left( \frac{\Delta G_{\text{dock}} - \Delta G_{\text{ref}}}{\tau} \right)}$$

donde:
- $\Delta G_{\text{ref}} = -6.0\text{ kcal/mol}$ corresponde a la afinidad umbral de reconocimiento micromolar ($K_d \approx 40\ \mu\mathrm{M}$ a $298\text{ K}$).
- $\tau = 1.5\text{ kcal/mol}$ define el ancho de transición suave de la escala sigmoidal.
- Valores más negativos que $-8.0\text{ kcal/mol}$ saturan asintóticamente $S_{\text{bind}} \to 1.0$, mientras que valores pobres ($\Delta G > -4.0\text{ kcal/mol}$) decaen a $S_{\text{bind}} \to 0.0$.

### 2.3 Equilibrio Tautomérico de Boltzmann

Cuando el ligando presenta tautomería en disolución acuosa, la población relativa de cada especie $i$ en equilibrio termodinámico se calcula a partir de las energías electrónicas corregidas por solvatación:

$$p_i = \frac{\exp\left(-\frac{\Delta E_i}{R T}\right)}{\sum_{j=1}^{M} \exp\left(-\frac{\Delta E_j}{R T}\right)}$$

donde $\Delta E_i = E_i - \min_j(E_j)$ es la energía relativa del tautómero $i$ calculada mediante DFT en agua, $R = 1.9872 \times 10^{-3}\text{ kcal/(mol}\cdot\text{K)}$ es la constante de los gases y $T = 298.15\text{ K}$.

---

## 3. Dinámica Molecular Clásica Solvatada (GROMACS)

### 3.1 Mecánica Clásica e Integración de Trayectorias

La evolución temporal del complejo proteína-ligando en disolución explícita se rige por las ecuaciones del movimiento de Newton para un sistema de $N$ átomos con masas $m_i$ y posiciones $\mathbf{r}_i$:

$$m_i \frac{d^2 \mathbf{r}_i(t)}{dt^2} = -\nabla_i V(\mathbf{r}_1, \mathbf{r}_2, \dots, \mathbf{r}_N)$$

La integración numérica temporal se resuelve mediante el algoritmo **Leap-Frog Verlet**, con un paso de integración temporal de $\Delta t = 2.0\text{ fs}$:

$$\mathbf{v}_i\left(t + \frac{\Delta t}{2}\right) = \mathbf{v}_i\left(t - \frac{\Delta t}{2}\right) + \frac{\mathbf{F}_i(t)}{m_i} \Delta t$$

$$\mathbf{r}_i(t + \Delta t) = \mathbf{r}_i(t) + \mathbf{v}_i\left(t + \frac{\Delta t}{2}\right) \Delta t$$

Los enlaces que involucran átomos de hidrógeno se restringen a sus distancias de equilibrio mediante el algoritmo matricial **LINCS** (Linear Constraint Solver), permitiendo un paso de $2\text{ fs}$ sin inestabilidad de resonancia de alta frecuencia.

### 3.2 Campos de Fuerza Moleculares

La función de energía potencial empírica $V(\mathbf{r})$ es un campo de fuerza clásico atomístico de dos componentes principales:

$$V(\mathbf{r}) = V_{\text{enlazado}}(\mathbf{r}) + V_{\text{no enlazado}}(\mathbf{r})$$

#### A. Términos Enlazados
$$V_{\text{enlazado}} = \sum_{\text{enlaces}} \frac{k_b}{2} (b - b_0)^2 + \sum_{\text{ángulos}} \frac{k_\theta}{2} (\theta - \theta_0)^2 + \sum_{\text{diedros}} \sum_n \frac{V_n}{2} [1 + \cos(n\phi - \gamma)] + \sum_{\text{impropios}} \frac{k_\xi}{2} (\xi - \xi_0)^2$$

- **Proteína:** **AMBER99SB-ILDN**, que incorpora potenciales torsionales re-optimizados a nivel ab initio de alta precisión para las cadenas laterales de Isoleucina (I), Leucina (L), Aspartato (D) y Asparagina (N).
- **Ligando:** **GAFF2** (General Amber Force Field 2), adecuado para pequeñas moléculas farmacóforas, heterociclos y warheads covalentes.

#### B. Términos No Enlazados
$$V_{\text{no enlazado}} = \sum_{i < j} \left( 4\epsilon_{ij} \left[ \left(\frac{\sigma_{ij}}{r_{ij}}\right)^{12} - \left(\frac{\sigma_{ij}}{r_{ij}}\right)^6 \right] + \frac{q_i q_j}{4\pi \varepsilon_0 \varepsilon_r r_{ij}} \right)$$

1. **Potencial de Lennard-Jones 12-6:** Modela la repulsión de intercambio de Pauli ($r^{-12}$) y la atracción dispersiva de van der Waals ($r^{-6}$). Los parámetros cruzados se derivan mediante las reglas de mezcla de Lorentz-Berthelot:
   $$\sigma_{ij} = \frac{\sigma_i + \sigma_j}{2}, \quad \epsilon_{ij} = \sqrt{\epsilon_i \epsilon_j}$$
   Se aplica un corte esférico (*cutoff*) a $1.0\text{ nm}$ con correcciones analíticas de dispersión para presión y energía.

2. **Electrostática de Coulomb y PME:**
   Las cargas parciales del ligando se obtienen mediante el protocolo semiempírico **AM1-BCC** (Austin Model 1 con correcciones Bond Charge Corrections), que reproduce el potencial electrostático mecanocuántico HF/6-31G*.
   
   En condiciones periódicas de contorno (PBC), la electrostática de largo alcance se calcula mediante el algoritmo **PME (Particle Mesh Ewald)**:
   $$V_{\text{coul}} = V_{\text{directo}} + V_{\text{recíproco}} + V_{\text{auto}}$$
   El término directo decae exponencialmente mediante la función complementaria de error $\text{erfc}(\beta r_{ij})/r_{ij}$, mientras que el espacio recíproco se calcula aplicando transformadas rápidas de Fourier (FFT) en una malla tridimensional de cuarto orden con espaciado de $0.12\text{ nm}$.

#### C. Solvatación Explícita e Iones
- El disolvente se modela con el modelo de agua **SPC/E** (Extended Simple Point Charge), que incluye una polarización media inducida en el dipolo del agua ($\mu = 2.35\text{ D}$).
- La caja de simulación es un prisma rómbico-dodecaédrico con una distancia mínima soluto-borde de $1.0\text{ nm}$.
- El sistema se neutraliza y saliniza a fuerza iónica fisiológica ($0.15\text{ M NaCl}$) reemplazando moléculas de agua por iones $\mathrm{Na}^+$ y $\mathrm{Cl}^-$.

### 3.3 Control Termodinámico (Ensamble NPT)

Las simulaciones de producción se ejecutan en el ensamble isotérmico-isobárico ($NPT$ a $300\text{ K}$ y $1.0\text{ bar}$):
- **Termostato:** Reescalamiento estocástico de velocidad de **Bussi-Donadio-Parrinello** (*canonical velocity rescaling*, $\tau_t = 0.1\text{ ps}$), el cual reproduce la distribución de velocidades de Maxwell-Boltzmann de un ensamble canónico riguroso.
- **Barostato:** Barostato isotrópico de **Parrinello-Rahman** ($\tau_p = 2.0\text{ ps}$, compresibilidad del agua $\kappa = 4.5 \times 10^{-5}\text{ bar}^{-1}$).

---

## 4. Agrupamiento de Trayectorias y Selección del Medioide

Para identificar la conformación representativa más poblada en el equilibrio dinámico sin sesgo de estructuras arbitrarias, SharK procesa la trayectoria mediante el algoritmo de agrupamiento conformacional de **Daura et al. (GROMOS clustering)**.

### 4.1 Matriz de RMSD Atómico

Para cada par de fotogramas $i$ y $j$ muestreados en la trayectoria (típicamente $N_{\text{frames}} \approx 400\text{--}1000$ tras equilibración), se realiza una superposición de mínimos cuadrados de traslación y rotación sobre los carbonos alfa ($\mathrm{C}_\alpha$) del bolsillo catalítico, evaluando el RMSD de coordenadas de átomos pesados del ligando:

$$\text{RMSD}_{ij} = \min_{\mathbf{R}, \mathbf{t}} \sqrt{\frac{1}{N_{\text{atoms}}} \sum_{k=1}^{N_{\text{atoms}}} \|\mathbf{r}_{k,i} - (\mathbf{R} \mathbf{r}_{k,j} + \mathbf{t})\|^2}$$

donde $\mathbf{R}$ es la matriz ortogonal de rotación y $\mathbf{t}$ es el vector de traslación óptimos obtenidos mediante el algoritmo de cuaterniones de Horn.

### 4.2 Algoritmo de Identificación de Clusters

Dado un radio de corte conformacional $R_{\text{cut}} = 1.5\text{ \AA}$:

1. Se construye la lista de vecinos para cada fotograma $i$:
   $$\mathcal{N}_i = \{ j \mid \text{RMSD}_{ij} \le R_{\text{cut}} \}$$
2. Se identifica el fotograma con el mayor número de vecinos $|\mathcal{N}_{m_1}| = \max_i |\mathcal{N}_i|$.
3. El fotograma $m_1$ se define como el **medioide representativo** del Cluster 1. El Cluster 1 comprende $m_1$ y todos los miembros de $\mathcal{N}_{m_1}$.
4. Se eliminan todos los miembros del Cluster 1 del conjunto global de fotogramas.
5. Se repiten los pasos 2 a 4 recursivamente para conformar el Cluster 2, Cluster 3, etc., hasta agotar la trayectoria.

El snapshot representativo del bolsillo extraído del medioide del Cluster 1 (el cluster dominante, típicamente $> 80\text{--}95\%$ de la población) se utiliza como el punto de partida estructural para los cálculos mecanocuánticos de sitio activo y estado de transición.

---

## 5. Pilar 2: Teoría de la Conformación de Pre-Ataque (Near-Attack Conformation, NAC)

### 5.1 El Concepto de Bruice de la Catálisis Enzimática

Propuesta por Thomas C. Bruice y colaboradores, la teoría de Near-Attack Conformation (NAC) establece que la catálisis enzimática y la aceleración de reacciones intramoleculares están gobernadas predominantemente por la fracción de tiempo en que los reactivos se encuentran atrapados en una subpoblación de geometrías conformacionales pre-organizadas donde:
1. La distancia entre el par de átomos que formará el enlace está dentro de la suma de sus radios de van der Waals más un incremento térmico permisible:
   $$d \le r_{\text{vdW}}(\mathrm{Nu}) + r_{\text{vdW}}(\mathrm{El}) + \delta \approx 3.2\text{--}3.5\text{ \AA}$$
2. El vector de ataque nucleofílico respecto al centro aceptor cumple la trayectoria orbital de Bürgi-Dunitz:
   $$\theta_{\text{BD}} \in [90^\circ, 135^\circ]$$

### 5.2 Geometría de Bürgi-Dunitz

En la adición nucleofílica sobre carbonilos, warheads vinílicos o grupos nitroso/aromáticos deficientes en electrones, el solapamiento óptimo ocurre cuando el orbital donador ataca el lóbulo $\pi^*$ antienlazante en un ángulo cercano a $\theta_{\text{BD}} \approx 105^\circ\text{--}107^\circ$.

El ángulo $\theta_{\text{BD}}$ en cada fotograma $t$ se define trigonométricamente mediante el producto escalar entre el vector de ataque nucleofílico $\mathbf{u}_{\text{att}}$ y el vector normal o de enlace del warhead $\mathbf{u}_{\text{warhead}}$:

$$\theta_{\text{BD}}(t) = \arccos\left( \frac{\mathbf{r}_{\text{Nu}}(t) - \mathbf{r}_{\text{El}}(t)}{\|\mathbf{r}_{\text{Nu}}(t) - \mathbf{r}_{\text{El}}(t)\|} \cdot \frac{\mathbf{r}_{\text{adj}}(t) - \mathbf{r}_{\text{El}}(t)}{\|\mathbf{r}_{\text{adj}}(t) - \mathbf{r}_{\text{El}}(t)\|} \right)$$

### 5.3 Métrica de Frecuencia Temporal ($P_{\text{NAC}}$)

A lo largo de toda la trayectoria temporal $t \in [0, T_{\text{sim}}]$, SharK evalúa la función indicadora de preorganización reactiva:

$$P_{\text{NAC}} = \frac{1}{N_{\text{frames}}} \sum_{t=1}^{N_{\text{frames}}} \mathbb{I}\left( d(t) \le 3.5\text{ \AA} \land \theta_{\text{BD}}(t) \in [90^\circ, 135^\circ] \right)$$

- Si $P_{\text{NAC}} \ge 0.05$ ($5\%$), la enzima preorganiza eficientemente la reacción y el inhibidor permanece listo para reaccionar.
- Si $P_{\text{NAC}} < 0.01$ ($1\%$), la barrera entrópica de aproximación en el bolsillo inhibe la reacción covalente a pesar de que el ligando tenga alta afinidad de docking.

### 5.4 Naturaleza Heurística de la Curva de Viabilidad de Bruice

En los reportes de SharK se grafica la *Bruice Near-Attack Feasibility Curve*:

$$f_{\text{Bruice}}(d) = \frac{1}{1 + \exp\left( \frac{d - d_0}{s} \right)}$$

donde $d_0 = 3.2\text{ \AA}$ y $s = 0.4\text{ \AA}$.

**Justificación del aviso metodológico ("Uncalibrated distance score / ranking heuristic"):**
Esta curva es una función de transferencia sigmoidal estandarizada en la literatura de modelado molecular para **ordenar por rango estático** diversas poses. No está ajustada analíticamente a la constante cinética específica de la reacción particular en estudio, puesto que una distancia geométrica de aproximación por sí sola no considera la estructura electrónica de la barrera de activación. La verdadera cinética cuantitativa de la reacción química proviene exclusivamente del Pilar 3 (Teoría del Estado de Transición con ORCA).

---

## 6. Mecánica Cuántica Molecular: Teoría del Funcional de la Densidad (DFT)

### 6.1 Ecuaciones Autoconsistentes de Kohn-Sham

En el formalismo de Kohn-Sham, el problema fundamental de electrones interactuantes se mapea exactamente en un sistema ficticio de electrones no interactuantes sometidos a un potencial efectivo $v_{\text{eff}}(\mathbf{r})$ que reproduce la densidad electrónica del estado fundamental $\rho(\mathbf{r})$:

$$\left[ -\frac{1}{2}\nabla^2 + v_{\text{eff}}(\mathbf{r}) \right] \psi_i(\mathbf{r}) = \epsilon_i \psi_i(\mathbf{r})$$

$$\rho(\mathbf{r}) = \sum_{i=1}^{N_{\text{occ}}} |\psi_i(\mathbf{r})|^2$$

donde el potencial efectivo se descompone en:
$$v_{\text{eff}}(\mathbf{r}) = v_{\text{ext}}(\mathbf{r}) + \int \frac{\rho(\mathbf{r}')}{\|\mathbf{r} - \mathbf{r}'\|} d\mathbf{r}' + v_{\text{xc}}[\rho](\mathbf{r})$$

siendo $v_{\text{xc}}[\rho](\mathbf{r}) = \frac{\delta E_{\text{xc}}[\rho]}{\delta \rho(\mathbf{r})}$ el potencial de intercambio y correlación.

### 6.2 El Funcional Compuesto r2SCAN-3c

Para modelar los sistemas moleculares y los clusters del sitio activo, SharK utiliza el método cuántico **r2SCAN-3c** (Grimme et al.), el cual combina:

1. **Funcional Meta-GGA r2SCAN:** Regulariza la densidad de energía cinética orbital positiva $\tau(\mathbf{r}) = \frac{1}{2}\sum_i |\nabla \psi_i|^2$ mediante el parámetro adimensional $\alpha = (\tau - \tau_{\text{von Weizsäcker}}) / \tau_{\text{Thomas-Fermi}}$, restaurando la respuesta uniforme y eliminando inestabilidades numéricas de integración en malla.
2. **Corrección de Dispersión D4:** Modela las fuerzas de van der Waals de largo alcance dependientes del entorno de enlace y las cargas parciales atómicas dinámicas mediante coeficientes de dispersión dependientes de la frecuencia $C_6^{ij}(\chi), C_8^{ij}$.
3. **Corrección de Error de Superposición de Base (gCP):** Evalúa analíticamente el *Basis Set Superposition Error* (BSSE) geométrico sin el elevado coste del formalismo tradicional de contrabalanceo de Boys-Bernardi.
4. **Conjunto de Base Modificado mTZVPP:** Base compacta optimizada con funciones de polarización que alcanza precisión cercana al límite de Hartree-Fock/Kohn-Sham completo con una fracción del coste computacional de bases convencionales cuádruple-zeta.

### 6.3 Modelo de Solvatación Dieléctrica Continuo (CPCM)

La polarización del disolvente acuoso alrededor del ligando y del cluster activo se incorpora mediante el modelo continuo **CPCM (Conductor-like Polarizable Continuum Model)**:
- Se genera una cavidad molecular alrededor del soluto basada en esferas atómicas unidas (radios de Bondi/Gavezotti aumentados en un factor de escala de $1.2$).
- La superficie de la cavidad se discretiza en pequeños teselas (*tesserae*).
- Se calculan las densidades de carga de polarización aparente inducida $\sigma(\mathbf{s})$ en cada elemento de superficie satisfaciendo las condiciones de contorno de un conductor perfecto ($\epsilon \to \infty$) escalado por el factor dieléctrico del agua ($\epsilon_{\text{agua}} = 80.4$ a $298.15\text{ K}$):

$$f(\epsilon) = \frac{\epsilon - 1}{\epsilon + x}$$

donde $x \approx 0.5$. La interacción entre la densidad electrónica molecular del soluto y las cargas de polarización de la cavidad se incluye de manera autoconsistente en cada ciclo SCF del operador de Fock.

---

## 7. Teoría de Orbitales de Frontera (FMO) y DFT Conceptual (CDFT)

### 7.1 Teorema de Koopmans y Descriptores Globales de Reactividad

A partir de los autovalores de energía orbital de Kohn-Sham del orbital molecular más alto ocupado ($\epsilon_{\text{HOMO}}$) y del orbital molecular más bajo desocupado ($\epsilon_{\text{LUMO}}$), la teoría de DFT Conceptual (Parr, Pearson, Yang) define rigurosamente los descriptores de reactividad electrónica química:

1. **Potencial de Ionización y Afinidad Electrónica:**
   $$I \approx -\epsilon_{\text{HOMO}}, \quad A \approx -\epsilon_{\text{LUMO}}$$

2. **Dureza Química ($\eta$):** Resistencia del sistema electrónico a deformar o transferir su nube de carga:
   $$\eta = \frac{I - A}{2} = \frac{\epsilon_{\text{LUMO}} - \epsilon_{\text{HOMO}}}{2} = \frac{\Delta \epsilon_{\text{gap}}}{2}$$

3. **Potencial Químico Electrónico ($\mu$):** Tendencia de los electrones a escapar del equilibrio electrónico:
   $$\mu = -\chi = -\frac{I + A}{2} = \frac{\epsilon_{\text{HOMO}} + \epsilon_{\text{LUMO}}}{2}$$
   donde $\chi$ es la electronegatividad absoluta de Mulliken.

4. **Índice de Electrofilia Global de Parr ($\omega$):** Capacidad termodinámica de un agente químico electrofílico para estabilizarse energéticamente al saturarse de densidad electrónica proveniente del entorno:
   $$\omega = \frac{\mu^2}{2\eta} = \frac{(\epsilon_{\text{HOMO}} + \epsilon_{\text{LUMO}})^2}{4(\epsilon_{\text{LUMO}} - \epsilon_{\text{HOMO}})}$$

5. **Suavidad Química ($S$):** Polarizabilidad molecular:
   $$S = \frac{1}{2\eta}$$

### 7.2 Funciones de Fukui Condensadas ($f_k^+$) y Regioespecificidad

La reactividad de cada átomo individual $k$ del ligando frente a un ataque nucleofílico (como el ataque del residuo `THR309:OG1`) está gobernada por la **Función de Fukui condensada para ataque nucleofílico ($f_k^+$)**:

$$f_k^+ = q_k(N+1) - q_k(N)$$

donde $q_k(N)$ y $q_k(N+1)$ son las poblaciones electrónicas atómicas (Mulliken o Hirshfeld) del sistema neutro y del anión mononegativo con geometría congelada.

El átomo del warhead que maximiza $f_k^+$ se identifica cuantitativamente como el centro electrofílico preferente (Rango #1 de regioespecificidad).

---

## 8. Pilar 3 y Tier 4: Modelado del Estado de Transición (TS) y Cinética de Eyring

### 8.1 Barrido Relajado de Coordenada de Reacción (ORCA Scan)

Para determinar el camino de mínima energía (MEP) de la adición covalente, se formula una optimización geométrica restringida sobre el cluster cuántico activo mediante multiplicadores de Lagrange. La función de Lagrange para cada paso de la coordenada $k$ es:

$$\mathcal{L}(\mathbf{R}, \lambda) = E_{\text{el}}(\mathbf{R}) + \frac{k_{\text{scan}}}{2} \left( \|\mathbf{r}_{\text{Nu}} - \mathbf{r}_{\text{El}}\| - d_{\text{scan}}^{(k)} \right)^2$$

A cada valor $d_{\text{scan}}^{(k)}$, todos los demás $3N-7$ grados de libertad nucleares se relajan libremente hasta satisfacer los criterios de convergencia geométrica:
- Cambio de energía $\Delta E < 5.0 \times 10^{-6}\text{ Eh}$.
- Gradiente máximo $\max_i \|\nabla_i E\| < 3.0 \times 10^{-4}\text{ Eh/bohr}$.
- Desplazamiento nuclear máximo $\Delta R_{\max} < 4.0 \times 10^{-3}\text{ bohr}$.

### 8.2 Perfil Energético y Estimación de la Barrera de Activación

El escaneo produce una serie discreta de energías electrónicas convergidas $\{ (d_k, E_k) \}_{k=1}^M$.
1. **Mínimo prerreactivo de reactivo:**
   $$E_{\text{min}} = \min_{k \le k_{\text{TS}}} E_k$$
   Ocurre a la distancia de equilibrio $d_{\text{min}} \approx 2.8\text{--}3.0\text{ \AA}$, donde las interacciones de van der Waals y enlaces de hidrógeno alcanzan su punto más estable antes de iniciar la repulsión de capas electrónicas.

2. **Cúspide del Estado de Transición ($E_{\text{TS}}$):**
   $$E_{\text{TS}} = \max_k E_k$$
   Ocurre a la distancia de punto de ensilladura estimada $d_{\text{TS}} \approx 2.1\text{--}2.3\text{ \AA}$.

3. **Barrera de Activación Electrónica:**
   $$\Delta E^\ddagger = E_{\text{TS}} - E_{\text{min}}$$
   Convertida a $\text{kcal/mol}$ mediante el factor de conversión exacto CODATA ($1\text{ Hartree} = 627.509474\text{ kcal/mol}$).

4. **Energía de Reacción del Aducto Covalente:**
   $$\Delta E_{\text{rxn}} = E_{\text{aducto}} - E_{\text{min}}$$
   donde $E_{\text{aducto}}$ es la energía en la cuenca de producto tras completarse el enlace ($d \approx 1.4\text{--}2.1\text{ \AA}$).

### 8.3 Teoría del Estado de Transición de Eyring-Polanyi

La constante de velocidad intrínseca de la etapa de modificación covalente ($k_{\text{chem}}$) en la vecindad del sitio activo se formula a través de la ecuación de Eyring-Polanyi de mecánica estadística cuántica:

$$k_{\text{chem}} = \kappa \frac{k_B T}{h} \exp\left( -\frac{\Delta G^\ddagger}{R T} \right)$$

donde:
- $k_B = 1.380649 \times 10^{-23}\text{ J/K}$ es la constante de Boltzmann.
- $h = 6.62607015 \times 10^{-34}\text{ J}\cdot\text{s}$ es la constante de Planck.
- A temperatura fisiológica ($T = 310.15\text{ K}$, $37^\circ\mathrm{C}$):
  $$\frac{k_B T}{h} \approx 6.4624 \times 10^{12}\text{ s}^{-1}$$
- $\kappa \approx 1.0$ es el coeficiente de transmisión adiabático.
- $\Delta G^\ddagger \approx \Delta E^\ddagger$ en la aproximación de barrido electrónico de cluster restringido con correcciones térmicas continuas.

El tiempo de vida media intrínseco de formación del aducto covalente es:

$$t_{1/2} = \frac{\ln 2}{k_{\text{chem}}}$$

### 8.4 Función de Transferencia Cinética ($S_{\text{chem}}$)

Para su integración en el índice de factibilidad global, la barrera de activación $\Delta G^\ddagger$ se mapea en una escala adimensional normalizada en $(0, 1]$ mediante una sigmoide logística inversa:

$$S_{\text{chem}} = \frac{1}{1 + \exp\left( \frac{\Delta G^\ddagger - \Delta G^\ddagger_{\text{midpoint}}}{\sigma_{\text{TS}}} \right)}$$

donde:
- $\Delta G^\ddagger_{\text{midpoint}} = 20.0\text{ kcal/mol}$ representa el umbral clásico de reactividad covalente biológicamente viable ($t_{1/2} \approx 1\text{ segundo}$).
- $\sigma_{\text{TS}} = 2.0\text{ kcal/mol}$ modula la pendiente de sensibilidad térmica.
- Barreras muy bajas ($\Delta G^\ddagger < 14\text{ kcal/mol}$) producen $S_{\text{chem}} > 0.95$ (cinética en microsegundos).
- Barreras prohibitivas ($\Delta G^\ddagger > 26\text{ kcal/mol}$) producen $S_{\text{chem}} < 0.05$ (inactivación inviable).

---

## 9. Integración Unificada: El Índice de Factibilidad Covalente (CFI)

Para formular un criterio unívoco de selección de fármacos covalentes, SharK integra los tres pilares independientes mediante una **media geométrica ponderada generalizada**:

$$\text{CFI}_{\text{final}} = \left( S_{\text{bind}} \right)^{w_{\text{bind}}} \cdot \left( P_{\text{NAC}} \right)^{w_{\text{nac}}} \cdot \left( S_{\text{chem}} \right)^{w_{\text{chem}}}$$

Calculado analíticamente en el espacio logarítmico para garantizar estabilidad numérica:

$$\ln(\text{CFI}_{\text{final}}) = w_{\text{bind}} \ln\left( \max(S_{\text{bind}}, \epsilon) \right) + w_{\text{nac}} \ln\left( \max(P_{\text{NAC}}, \epsilon) \right) + w_{\text{chem}} \ln\left( \max(S_{\text{chem}}, \epsilon) \right)$$

$$\text{CFI}_{\text{final}} = \exp\left( \ln(\text{CFI}_{\text{final}}) \right)$$

donde $\epsilon = 10^{-12}$ previene singularidades asintóticas y los pesos relativos predeterminados satisfacen la condición de normalización:

$$w_{\text{bind}} + w_{\text{nac}} + w_{\text{chem}} = 0.20 + 0.40 + 0.40 = 1.00$$

### 9.1 Índice Prerreactivo Provisional ($\text{CFI}_{\text{pre}}$)

Cuando el cálculo de estado de transición del Pilar 3 aún no se ha ejecutado, SharK calcula el índice prerreactivo de reconocimiento dinámico reescalando los dos primeros pilares:

$$\text{CFI}_{\text{pre}} = \left( S_{\text{bind}} \right)^{\frac{w_{\text{bind}}}{w_{\text{bind}} + w_{\text{nac}}}} \cdot \left( P_{\text{NAC}} \right)^{\frac{w_{\text{nac}}}{w_{\text{bind}} + w_{\text{nac}}}} = \left( S_{\text{bind}} \right)^{0.333} \cdot \left( P_{\text{NAC}} \right)^{0.667}$$

### 9.2 Taxonomía de Factibilidad Covalente

Conforme al valor final de $\text{CFI}_{\text{final}}$, el inhibidor se categoriza en:

| Rango de $\text{CFI}_{\text{final}}$ | Categoría | Interpretación Farmacológica |
| :---: | :---: | :--- |
| $\text{CFI}_{\text{final}} \ge 0.70$ | **High Covalent Feasibility** | Excelente reconocimiento reversible, alta persistencia de pre-ataque y barrera cuántica fácilmente superable. Candidato prioritario. |
| $0.40 \le \text{CFI}_{\text{final}} < 0.70$ | **Moderate Covalent Feasibility** | Fármaco viable pero susceptible de optimización (ej. buen reconocimiento pero orientación geométrica subóptima). |
| $\text{CFI}_{\text{final}} < 0.40$ | **Low Covalent Feasibility** | Cuello de botella severo en al menos uno de los tres pilares (poca afinidad, alejamiento en la dinámica molecular o barrera química prohibitiva). |

---

## 10. Matriz de Invariantes Científicos y Auditoría de Control de Calidad

Para asegurar la reproducibilidad, rigor físico y honestidad epistemológica del software, el motor de SharK valida formalmente en tiempo de ejecución los siguientes invariantes:

- **INV-001:** Las poses de acoplamiento estático tienen **estrictamente prohibido** poblar o fingir la probabilidad dinámica $P_{\text{NAC}}$.
- **INV-002:** El cálculo de $\text{CFI}_{\text{pre}}$ requiere obligatoriamente la trayectoria solvatada de dinámica molecular para derivar $P_{\text{NAC}}$. Si no hay dinámica, $\text{CFI}_{\text{pre}} = \text{None}$.
- **INV-003:** El índice $\text{CFI}_{\text{final}}$ exige sin excepciones la existencia de los Tres Pilares ($S_{\text{bind}}$, $P_{\text{NAC}}$ y $S_{\text{chem}}$). Si falta la barrera cuántica $\Delta G^\ddagger$, $\text{CFI}_{\text{final}}$ no se calcula y se reporta en estado pendiente.
- **INV-004:** SharK tiene prohibido recurrir a puntuaciones geométricas de contacto estático como sustituto de $\text{CFI}_{\text{final}}$.
- **INV-007 / INV-010:** En estructuras prerreactivas ($d > 2.0\text{ \AA}$), el software tiene **prohibido calcular o inventar órdenes de enlace de Wiberg (WBO) o transferencias de carga ($\Delta q$)** del producto covalente. Dichos parámetros solo son válidos para aductos optimizados donde el enlace ya está formado.
- **INV-012:** Toda propiedad derivada de geometrías estáticas o modelos empíricos debe llevar explícitamente la insignia epistemológica `[Model-Derived Proxy]`, advirtiendo al investigador que se trata de una aproximación heurística y no de un observable cuántico ab initio directo.

---

## 11. Constantes Físicas y Factores de Conversión

| Constante / Magnitud | Símbolo | Valor Numérico | Unidades |
| :--- | :---: | :---: | :--- |
| Constante de Planck | $h$ | $6.62607015 \times 10^{-34}$ | $\mathrm{J}\cdot\mathrm{s}$ |
| Constante de Boltzmann | $k_B$ | $1.380649 \times 10^{-23}$ | $\mathrm{J}\cdot\mathrm{K}^{-1}$ |
| Constante Universal de los Gases | $R$ | $1.9872042586 \times 10^{-3}$ | $\mathrm{kcal}\cdot\mathrm{mol}^{-1}\cdot\mathrm{K}^{-1}$ |
| Hartree a Kilocalorías por Mol | — | $627.509474$ | $\mathrm{kcal}\cdot\mathrm{mol}^{-1}\cdot\mathrm{Hartree}^{-1}$ |
| Hartree a Electronvoltios | — | $27.2113862$ | $\mathrm{eV}\cdot\mathrm{Hartree}^{-1}$ |
| Electronvoltio a Kilocalorías por Mol | — | $23.0605489$ | $\mathrm{kcal}\cdot\mathrm{mol}^{-1}\cdot\mathrm{eV}^{-1}$ |
| Radio de Bohr a Ångström | $a_0$ | $0.5291772109$ | $\mathrm{\AA}\cdot\mathrm{bohr}^{-1}$ |
| Velocidad de la Luz en el Vacío | $c$ | $2.99792458 \times 10^{10}$ | $\mathrm{cm}\cdot\mathrm{s}^{-1}$ |
| Factor Prefactorial de Eyring ($310.15\text{ K}$) | $k_B T / h$ | $6.4624 \times 10^{12}$ | $\mathrm{s}^{-1}$ |
