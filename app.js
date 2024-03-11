async function initialize() {
    let pyodide = await loadPyodide({
        indexURL: "https://cdn.jsdelivr.net/pyodide/v0.18.1/full/"
    });
    await pyodide.loadPackage('numpy');

    await pyodide.runPythonAsync(`
    import numpy as np
    
    class GridWorld:
        def __init__(self, n_rows, n_cols, goal, bad, wall, gamma=0.9):
            self.n_rows = n_rows
            self.n_cols = n_cols
            self.goal = goal
            self.bad = bad
            self.wall = wall
            self.gamma = gamma  # Initialize the discount factor
            self.actions = [(1, 0), (-1, 0), (0, -1), (0, 1)]
            self.V = np.zeros((n_rows, n_cols))
            self.V[self.goal] = 1
            self.V[self.bad] = -1
            self.reward = np.zeros((n_rows, n_cols))  # Initialize the rewards matrix
            # Assign rewards for specific states
            self.reward[self.goal] = 1
            self.reward[self.bad] = -1
        
        def move(self, state, action):
            x, y = state
            if action == (1, 0):
                return (x + 1, y)
            elif action == (-1, 0):
                return (x - 1, y)
            elif action == (0, 1):
                return (x, y + 1)
            elif action == (0, -1):
                return (x, y - 1)
    
        def is_valid_move(self, position, action):
            new_position = self.move(position, action)
            x, y = new_position
            if 0 <= x < self.n_rows and 0 <= y < self.n_cols and new_position != self.wall:
                return True
            return False
    
        def possible_actions(self, position):
            valid_actions = []
            if position != self.wall:
                for action in self.actions:
                    if self.is_valid_move(position, action):
                        valid_actions.append(action)
            return valid_actions
    
        def action_90degree(self, action, actions_possible):
            actions = []
            for i in actions_possible:
                if i != action:
                    if i + action != (0, 0):
                        actions.append(i)
            return actions
    
        def get_state_index(self, x, y):
            return x * self.n_cols + y

            
        def iterate_value(self):
            delta = 0
            new_V = np.zeros((self.n_rows, self.n_cols))
            
            for x in range(self.n_rows):
                for y in range(self.n_cols):
                    if (x, y) == self.wall or (x, y) == self.goal or (x, y) == self.bad:
                        new_V[x, y] = self.V[x, y]  # Conservez les valeurs pour le mur, l'objectif et les cases pénalisantes
                        continue
                    
                    Vs = []
                    pa = self.possible_actions((x, y))
                    for action in pa:
                        sum_derivations = 0
                        for other_action in self.action_90degree(action, pa):
                            proba = 0.2 if len(self.action_90degree(action, pa)) == 1 else 0.1
                            move_result = self.move((x, y), other_action)
                            if self.is_valid_move((x, y), other_action):
                                sum_derivations += proba * self.V[move_result]
                        
                        move_result = self.move((x, y), action)
                        if self.is_valid_move((x, y), action):
                            sum_action = self.reward[x, y] + self.gamma * 0.8 * self.V[move_result]
                            sum_rewards = sum_action + sum_derivations
                            Vs.append(sum_rewards)
                    
                    new_V[x, y] = max(Vs) if Vs else self.V[x, y]  # Mettez à jour avec la valeur maximale ou conservez l'ancienne si Vs est vide
            
            self.V = new_V  # Mettez à jour la matrice des valeurs
            return np.max(np.abs(self.V - new_V))  # Retournez le changement maximal pour vérifier la convergence
        
        `);
    // Ensuite, créez une instance de GridWorld dans un appel séparé
    await pyodide.runPythonAsync(`
    gridWorld = GridWorld(3, 4, (0, 3), (1, 3), (1, 1))
    `);


    function createGrid(n_rows, n_cols) {
        const gridContainer = document.getElementById('grid-container');
        if (!gridContainer) {
            console.error('Grid container not found');
            return;
        }

        gridContainer.innerHTML = ''; // Effacez les cellules précédentes si nécessaire
    
        for (let x = 0; x < n_rows; x++) {
            for (let y = 0; y < n_cols; y++) {
                const cell = document.createElement('div');
                cell.classList.add('grid-cell');
                cell.id = `cell-${x}-${y}`; 
                cell.style.width = '50px'; // Taille des cellules, ajustez selon vos besoins
                cell.style.height = '50px';
                cell.style.border = '1px solid black';
                cell.style.display = 'inline-block'; // ou "flex"
                gridContainer.appendChild(cell);
            }
            // Ajoutez une nouvelle ligne après chaque rangée si nécessaire
            const breakLine = document.createElement('div');
            breakLine.style.clear = 'both';
            gridContainer.appendChild(breakLine);
        }
    }

    async function updateGridDisplay() {
        let gridValuesPython = await pyodide.runPythonAsync('gridWorld.V.tolist()');
        console.log("Grid values from Python:", gridValuesPython);
        let gridValues = gridValuesPython.toJs();
        console.log("Grid values converted to JS:", gridValues);

        if (!gridValues || !Array.isArray(gridValues)) {
            console.error("gridValues is undefined or not an array");
            return;
        }
        
            
        for (let x = 0; x < gridValues.length; x++) {
            for (let y = 0; y < gridValues[x].length; y++) {
                const cellId = `cell-${x}-${y}`;
                const cell = document.getElementById(cellId);
                if (cell) {
                    cell.textContent = gridValues[x][y].toFixed(2);
                } else {
                    console.error('Cell not found:', cellId);
                }
            }
        }
    }
            
    // Initialisation de la grille dans l'interface utilisateur
    createGrid(3, 4); // Assurez-vous que cette fonction est définie pour correspondre à la structure de votre grille
    let iterationCount = 0; // Initialize iteration count

    // Fonction pour mettre à jour l'affichage de la grille basée sur l'état de GridWorld.V
    // Mise à jour du nombre d'itérations dans l'UI
    function updateIterationCount() {
        document.getElementById('iterationCount').textContent = `Nombre d'itérations : ${iterationCount}`;
    }

    document.getElementById('iterateBtn').addEventListener('click', async () => {
        const delta = await pyodide.runPythonAsync('gridWorld.iterate_value()');
        await updateGridDisplay(pyodide);
        iterationCount += 1; // Incrémentation du compteur d'itérations
        updateIterationCount();
        updateGridDisplay(pyodide);
    
    });
    document.getElementById('resetBtn').addEventListener('click', async () => {
        iterationCount = 0; // Réinitialisation du compteur
        updateIterationCount();
            await pyodide.runPythonAsync(`
            gridWorld = GridWorld(3, 4, (0, 3), (1, 3), (1, 1))
        `);
        await updateGridDisplay(pyodide); // Update grid display to reflect reset state
    });
    
}
    
initialize();