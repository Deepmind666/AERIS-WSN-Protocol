# 娣峰悎鍏冨惎鍙戝紡绠楁硶妯″潡 - 鐢ㄤ簬AERIS鍗忚
# 鍙傝€冩枃鐚細
# [4] Kuila, P., & Jana, P. K. (2014). Energy efficient clustering and routing algorithms for wireless sensor networks: Particle swarm optimization approach.
# [5] Elhabyan, R. S., & Yagoub, M. C. (2015). Two-tier particle swarm optimization protocol for clustering and routing in wireless sensor network.
# [6] Singh, B., & Lobiyal, D. K. (2012). A novel energy-aware cluster head selection based on particle swarm optimization for wireless sensor networks.
# [7] Gupta, G. P., & Jha, S. (2018). Integrated clustering and routing protocol for wireless sensor networks using Cuckoo and harmony search based metaheuristic techniques.

import numpy as np
import random
import math

class HybridMetaheuristic:
    """娣峰悎鍏冨惎鍙戝紡绠楁硶锛岀粨鍚圥SO鍜孉CO浼樺寲WSN璺敱"""
    
    def __init__(self, network_size=100, area_size=200, base_station_pos=(100, 100)):
        self.network_size = network_size
        self.area_size = area_size
        self.base_station = base_station_pos
        
        # PSO鍙傛暟
        self.pso_population_size = 20
        self.pso_max_iterations = 50
        self.w = 0.7  # 鎯€ф潈閲?
        self.c1 = 1.5  # 璁ょ煡鍙傛暟
        self.c2 = 1.5  # 绀句細鍙傛暟
        
        # ACO鍙傛暟
        self.aco_ants = 20
        self.aco_iterations = 30
        self.alpha = 1.0  # 淇℃伅绱犻噸瑕佹€?
        self.beta = 2.0   # 鍚彂寮忎俊鎭噸瑕佹€?
        self.rho = 0.1    # 淇℃伅绱犺捀鍙戠巼
        self.Q = 100      # 淇℃伅绱犲己搴?
        
        # 娣峰悎绠楁硶鍙傛暟
        self.hybrid_iterations = 10
        self.pso_weight = 0.5  # PSO缁撴灉鏉冮噸
        self.aco_weight = 0.5  # ACO缁撴灉鏉冮噸
    
    def optimize_clustering(self, nodes, n_clusters):
        """浣跨敤PSO浼樺寲缃戠粶鍒嗙皣
        
        鍙傛暟:
            nodes: 缃戠粶鑺傜偣鍒楄〃
            n_clusters: 鏈熸湜鐨勭皣鏁伴噺
            
        杩斿洖:
            鏈€浼樼殑绨囧ご鑺傜偣鍒楄〃
        """
        alive_nodes = [node for node in nodes if node.is_alive]
        if len(alive_nodes) <= n_clusters:
            return alive_nodes  # 濡傛灉娲昏穬鑺傜偣鏁板皯浜庣皣鏁帮紝鎵€鏈夎妭鐐归兘鏄皣澶?
        
        # 鍒濆鍖朠SO绮掑瓙缇?
        particles = []
        velocities = []
        personal_best_positions = []
        personal_best_scores = []
        
        # 鍏ㄥ眬鏈€浼樿В
        global_best_position = None
        global_best_score = -float('inf')
        
        # 鍒濆鍖栫矑瀛愪綅缃拰閫熷害
        for _ in range(self.pso_population_size):
            # 闅忔満閫夋嫨n_clusters涓妭鐐逛綔涓虹皣澶?
            particle = random.sample(range(len(alive_nodes)), n_clusters)
            particles.append(particle)
            
            # 鍒濆鍖栭€熷害涓?
            velocity = [0] * n_clusters
            velocities.append(velocity)
            
            # 璇勪及绮掑瓙閫傚簲搴?
            score = self._evaluate_clustering(particle, alive_nodes)
            
            # 鏇存柊涓綋鏈€浼?
            personal_best_positions.append(particle.copy())
            personal_best_scores.append(score)
            
            # 鏇存柊鍏ㄥ眬鏈€浼?
            if score > global_best_score:
                global_best_score = score
                global_best_position = particle.copy()
        
        # PSO杩唬浼樺寲
        for _ in range(self.pso_max_iterations):
            for i in range(self.pso_population_size):
                # 鏇存柊閫熷害鍜屼綅缃?
                for j in range(n_clusters):
                    # 鏇存柊閫熷害
                    r1, r2 = random.random(), random.random()
                    velocities[i][j] = (self.w * velocities[i][j] + 
                                       self.c1 * r1 * (personal_best_positions[i][j] - particles[i][j]) + 
                                       self.c2 * r2 * (global_best_position[j] - particles[i][j]))
                    
                    # 鏇存柊浣嶇疆锛堢鏁SO锛?
                    new_pos = particles[i][j] + int(velocities[i][j])
                    new_pos = max(0, min(new_pos, len(alive_nodes) - 1))
                    particles[i][j] = new_pos
                
                # 纭繚娌℃湁閲嶅鐨勭皣澶?
                particles[i] = list(set(particles[i]))
                while len(particles[i]) < n_clusters:
                    new_node = random.randint(0, len(alive_nodes) - 1)
                    if new_node not in particles[i]:
                        particles[i].append(new_node)
                
                # 璇勪及鏂颁綅缃?
                score = self._evaluate_clustering(particles[i], alive_nodes)
                
                # 鏇存柊涓綋鏈€浼?
                if score > personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = particles[i].copy()
                
                # 鏇存柊鍏ㄥ眬鏈€浼?
                if score > global_best_score:
                    global_best_score = score
                    global_best_position = particles[i].copy()
        
        # 杩斿洖鏈€浼樼皣澶?
        return [alive_nodes[idx] for idx in global_best_position]
    
    def _evaluate_clustering(self, particle, nodes):
        """璇勪及鍒嗙皣鏂规鐨勯€傚簲搴?
        
        鍙傛暟:
            particle: 琛ㄧず绨囧ご鐨勭矑瀛?
            nodes: 鎵€鏈夋椿璺冭妭鐐?
            
        杩斿洖:
            閫傚簲搴﹀緱鍒?
        """
        # 鑾峰彇绨囧ご鑺傜偣
        cluster_heads = [nodes[idx] for idx in particle]
        
        # 璁＄畻姣忎釜鑺傜偣鍒版渶杩戠皣澶寸殑璺濈
        total_distance = 0
        for node in nodes:
            if node not in cluster_heads:
                min_distance = float('inf')
                for ch in cluster_heads:
                    distance = node.calculate_distance(ch)
                    if distance < min_distance:
                        min_distance = distance
                total_distance += min_distance
        
        # 璁＄畻绨囧ご鐨勫钩鍧囪兘閲?
        avg_energy = sum(ch.energy for ch in cluster_heads) / len(cluster_heads)
        
        # 璁＄畻绨囧ご鐨勫钩鍧囪窛绂诲埌鍩虹珯
        avg_distance_to_bs = sum(ch.distance_to_bs for ch in cluster_heads) / len(cluster_heads)
        
        # 閫傚簲搴﹀嚱鏁帮細鏈€灏忓寲鑺傜偣鍒扮皣澶寸殑璺濈锛屾渶澶у寲绨囧ご鑳介噺锛屾渶灏忓寲绨囧ご鍒板熀绔欑殑璺濈
        # 褰掍竴鍖栧悇椤规寚鏍?
        norm_distance = 1 / (1 + total_distance / len(nodes))
        norm_energy = avg_energy / 2.0  # 鍋囪鍒濆鑳介噺涓?.0J
        norm_bs_distance = 1 / (1 + avg_distance_to_bs / 300)  # 鍋囪鏈€澶ц窛绂讳负300m
        
        # 缁煎悎閫傚簲搴?
        fitness = 0.4 * norm_distance + 0.4 * norm_energy + 0.2 * norm_bs_distance
        
        return fitness
    
    def optimize_routing(self, cluster_heads, base_station):
        """浣跨敤ACO浼樺寲绨囧ご鍒板熀绔欑殑璺敱
        
        鍙傛暟:
            cluster_heads: 绨囧ご鑺傜偣鍒楄〃
            base_station: 鍩虹珯鑺傜偣
            
        杩斿洖:
            浼樺寲鍚庣殑璺敱璺緞锛堟瘡涓皣澶寸殑涓嬩竴璺籌級
        """
        n_ch = len(cluster_heads)
        if n_ch == 0:
            return {}
        
        # 濡傛灉鍙湁涓€涓皣澶达紝鐩存帴杩炴帴鍒板熀绔?
        if n_ch == 1:
            return {cluster_heads[0].id: base_station}
        
        # 鏋勫缓璺濈鐭╅樀鍜屼俊鎭礌鐭╅樀
        # 娣诲姞鍩虹珯浣滀负鏈€鍚庝竴涓妭鐐?
        all_nodes = cluster_heads + [base_station]
        n_nodes = len(all_nodes)
        
        # 璺濈鐭╅樀
        distances = np.zeros((n_nodes, n_nodes))
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    distances[i, j] = all_nodes[i].calculate_distance(all_nodes[j])
                else:
                    distances[i, j] = float('inf')  # 鑷繁鍒拌嚜宸辩殑璺濈璁句负鏃犵┓澶?
        
        # 鍒濆鍖栦俊鎭礌鐭╅樀
        pheromones = np.ones((n_nodes, n_nodes)) * 0.1
        
        # 鍒濆鍖栨渶浣宠矾寰?
        best_path = None
        best_path_length = float('inf')
        
        # ACO杩唬浼樺寲
        for _ in range(self.aco_iterations):
            # 姣忓彧铓傝殎鏋勫缓璺緞
            ant_paths = []
            ant_path_lengths = []
            
            for _ in range(self.aco_ants):
                # 闅忔満閫夋嫨璧峰绨囧ご
                current = random.randint(0, n_ch - 1)
                path = [current]
                path_length = 0
                visited = [False] * n_nodes
                visited[current] = True
                
                # 鏋勫缓璺緞鐩村埌鎵€鏈夌皣澶撮兘琚闂?
                while len(path) < n_ch:
                    # 璁＄畻杞Щ姒傜巼
                    probabilities = np.zeros(n_nodes)
                    for j in range(n_ch):
                        if not visited[j]:
                            # 璁＄畻鍚彂寮忎俊鎭紙璺濈鐨勫€掓暟锛?
                            eta = 1.0 / distances[current, j]
                            # 璁＄畻杞Щ姒傜巼
                            probabilities[j] = (pheromones[current, j] ** self.alpha) * (eta ** self.beta)
                    
                    # 褰掍竴鍖栨鐜?
                    if np.sum(probabilities) > 0:
                        probabilities = probabilities / np.sum(probabilities)
                    
                    # 杞洏璧岄€夋嫨涓嬩竴涓妭鐐?
                    next_node = self._roulette_wheel_selection(probabilities)
                    
                    # 鏇存柊璺緞
                    path.append(next_node)
                    path_length += distances[current, next_node]
                    visited[next_node] = True
                    current = next_node
                
                # 杩炴帴鍒板熀绔?
                path.append(n_nodes - 1)  # 鍩虹珯绱㈠紩
                path_length += distances[current, n_nodes - 1]
                
                ant_paths.append(path)
                ant_path_lengths.append(path_length)
                
                # 鏇存柊鏈€浣宠矾寰?
                if path_length < best_path_length:
                    best_path_length = path_length
                    best_path = path.copy()
            
            # 鏇存柊淇℃伅绱?
            # 淇℃伅绱犺捀鍙?
            pheromones = (1 - self.rho) * pheromones
            
            # 淇℃伅绱犳矇绉?
            for i, path in enumerate(ant_paths):
                delta = self.Q / ant_path_lengths[i]
                for j in range(len(path) - 1):
                    pheromones[path[j], path[j+1]] += delta
        
        # 鏋勫缓璺敱琛紙姣忎釜绨囧ご鐨勪笅涓€璺籌級
        routing_table = {}
        
        # 浣跨敤鏈€浣宠矾寰勬瀯寤鸿矾鐢辫〃
        if best_path:
            # 灏嗚矾寰勮浆鎹负璺敱琛?
            for i in range(len(best_path) - 1):
                ch_id = cluster_heads[best_path[i]].id
                next_hop = all_nodes[best_path[i+1]]
                routing_table[ch_id] = next_hop
        
        return routing_table
    
    def _roulette_wheel_selection(self, probabilities):
        """杞洏璧岄€夋嫨
        
        鍙傛暟:
            probabilities: 姒傜巼鏁扮粍
            
        杩斿洖:
            閫変腑鐨勭储寮?
        """
        r = random.random()
        c = 0
        for i, p in enumerate(probabilities):
            c += p
            if r <= c:
                return i
        return len(probabilities) - 1
    
    def hybrid_optimize(self, nodes, n_clusters, base_station):
        """娣峰悎PSO鍜孉CO杩涜鍒嗙皣鍜岃矾鐢变紭鍖?
        
        鍙傛暟:
            nodes: 缃戠粶鑺傜偣鍒楄〃
            n_clusters: 鏈熸湜鐨勭皣鏁伴噺
            base_station: 鍩虹珯鑺傜偣
            
        杩斿洖:
            (cluster_heads, routing_table): 绨囧ご鍒楄〃鍜岃矾鐢辫〃
        """
        best_cluster_heads = None
        best_routing_table = None
        best_score = -float('inf')
        
        for _ in range(self.hybrid_iterations):
            # PSO浼樺寲鍒嗙皣
            cluster_heads = self.optimize_clustering(nodes, n_clusters)
            
            # ACO浼樺寲璺敱
            routing_table = self.optimize_routing(cluster_heads, base_station)
            
            # 璇勪及鏁翠綋瑙ｅ喅鏂规
            score = self._evaluate_solution(cluster_heads, routing_table, nodes, base_station)
            
            # 鏇存柊鏈€浣宠В鍐虫柟妗?
            if score > best_score:
                best_score = score
                best_cluster_heads = cluster_heads.copy()
                best_routing_table = routing_table.copy()
        
        return best_cluster_heads, best_routing_table
    
    def _evaluate_solution(self, cluster_heads, routing_table, nodes, base_station):
        """璇勪及鏁crete綋瑙ｅ喅鏂规鐨勮川閲?
        
        鍙傛暟:
            cluster_heads: 绨囧ご鍒楄〃
            routing_table: 璺敱琛?
            nodes: 鎵€鏈夎妭鐐?
            base_station: 鍩虹珯鑺傜偣
            
        杩斿洖:
            瑙ｅ喅鏂规寰楀垎
        """
        # 璇勪及鍒嗙皣璐ㄩ噺
        clustering_score = self._evaluate_clustering_solution(cluster_heads, nodes)
        
        # 璇勪及璺敱璐ㄩ噺
        routing_score = self._evaluate_routing_solution(routing_table, cluster_heads, base_station)
        
        # 缁煎悎寰楀垎
        return self.pso_weight * clustering_score + self.aco_weight * routing_score
    
    def _evaluate_clustering_solution(self, cluster_heads, nodes):
        """璇勪及鍒嗙皣瑙ｅ喅鏂规"""
        # 璁＄畻姣忎釜鑺傜偣鍒版渶杩戠皣澶寸殑骞冲潎璺濈
        total_distance = 0
        for node in nodes:
            if node.is_alive and node not in cluster_heads:
                min_distance = float('inf')
                for ch in cluster_heads:
                    distance = node.calculate_distance(ch)
                    if distance < min_distance:
                        min_distance = distance
                total_distance += min_distance
        
        avg_distance = total_distance / max(1, len(nodes) - len(cluster_heads))
        
        # 璁＄畻绨囧ご鐨勫钩鍧囪兘閲?
        avg_energy = sum(ch.energy for ch in cluster_heads) / max(1, len(cluster_heads))
        
        # 褰掍竴鍖?
        norm_distance = 1 / (1 + avg_distance / 100)  # 鍋囪鏈€澶ц窛绂讳负100m
        norm_energy = avg_energy / 2.0  # 鍋囪鍒濆鑳介噺涓?.0J
        
        # 缁煎悎寰楀垎
        return 0.5 * norm_distance + 0.5 * norm_energy
    
    def _evaluate_routing_solution(self, routing_table, cluster_heads, base_station):
        """璇勪及璺敱瑙ｅ喅鏂规"""
        if not routing_table or not cluster_heads:
            return 0
        
        # 璁＄畻璺敱璺緞鐨勬€婚暱搴?
        total_length = 0
        hop_count = 0
        
        for ch in cluster_heads:
            current = ch
            path_length = 0
            hops = 0
            
            # 璺熻釜璺緞鐩村埌鍒拌揪鍩虹珯鎴栨棤娉曠户缁?
            while current != base_station and current.id in routing_table:
                next_hop = routing_table[current.id]
                path_length += current.calculate_distance(next_hop)
                current = next_hop
                hops += 1
                
                # 闃叉寰幆
                if hops > len(cluster_heads):
                    break
            
            total_length += path_length
            hop_count += hops
        
        # 璁＄畻骞冲潎璺緞闀垮害鍜岃烦鏁?
        avg_length = total_length / len(cluster_heads)
        avg_hops = hop_count / len(cluster_heads)
        
        # 褰掍竴鍖?
        norm_length = 1 / (1 + avg_length / 300)  # 鍋囪鏈€澶ч暱搴︿负300m
        norm_hops = 1 / (1 + avg_hops / 10)  # 鍋囪鏈€澶ц烦鏁颁负10
        
        # 缁煎悎寰楀垎
        return 0.7 * norm_length + 0.3 * norm_hops

# 娴嬭瘯浠ｇ爜
if __name__ == "__main__":
    # 妯℃嫙鑺傜偣绫?
    class TestNode:
        def __init__(self, id, x, y, energy=2.0):
            self.id = id
            self.x = x
            self.y = y
            self.energy = energy
            self.is_alive = True
            self.distance_to_bs = math.sqrt((x - 100)**2 + (y - 100)**2)
        
        def calculate_distance(self, node):
            return math.sqrt((self.x - node.x)**2 + (self.y - node.y)**2)
    
    # 鍒涘缓娴嬭瘯鑺傜偣
    test_nodes = []
    for i in range(100):
        x = random.uniform(0, 200)
        y = random.uniform(0, 200)
        energy = random.uniform(0.5, 2.0)
        test_nodes.append(TestNode(i, x, y, energy))
    
    # 鍒涘缓鍩虹珯
    base_station = TestNode(-1, 100, 100, float('inf'))
    
    # 鍒涘缓娣峰悎鍏冨惎鍙戝紡绠楁硶瀹炰緥
    hybrid = HybridMetaheuristic()
    
    # 娴嬭瘯鍒嗙皣浼樺寲
    cluster_heads = hybrid.optimize_clustering(test_nodes, 5)
    print(f"Cluster heads after optimization: {len(cluster_heads)}")
    print(f"CH IDs: {[ch.id for ch in cluster_heads]}")
    
    # 娴嬭瘯璺敱浼樺寲
    routing_table = hybrid.optimize_routing(cluster_heads, base_station)
    print(f"Routing table size: {len(routing_table)}")
    print(f"CH {ch_id} -> next hop {next_hop.id}")
    
    # 娴嬭瘯娣峰悎浼樺寲
    best_chs, best_routes = hybrid.hybrid_optimize(test_nodes, 5, base_station)
    print(f"Merged-optimized CH count: {len(best_chs)}")
    print(f"Merged-optimized routing table size: {len(best_routes)}")
