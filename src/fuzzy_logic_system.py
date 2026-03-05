# 妯＄硦閫昏緫绯荤粺妯″潡 - 鐢ㄤ簬AERIS鍗忚
# 鍙傝€冩枃鐚細
# [1] Nayak, P., & Vathasavai, B. (2017). Energy efficient clustering algorithm for multi-hop wireless sensor network using type-2 fuzzy logic.
# [2] Balakrishnan, B., & Balachandran, S. (2017). FLECH: fuzzy logic based energy efficient clustering hierarchy for nonuniform wireless sensor networks.
# [3] Logambigai, R., & Kannan, A. (2018). Fuzzy logic based unequal clustering for wireless sensor networks.

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class FuzzyLogicSystem:
    """瀹屾暣鐨勬ā绯婇€昏緫绯荤粺锛岀敤浜嶹SN涓殑鍐崇瓥浼樺寲"""
    
    def __init__(self):
        # 鍒涘缓妯＄硦閫昏緫鎺у埗绯荤粺
        self.setup_fuzzy_system()
        
    def setup_fuzzy_system(self):
        """璁剧疆妯＄硦閫昏緫绯荤粺鐨勮緭鍏ャ€佽緭鍑哄拰瑙勫垯"""
        # 鍒涘缓妯＄硦鍙橀噺 - 杈撳叆
        self.residual_energy = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'residual_energy')
        self.node_centrality = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'node_centrality')
        self.node_degree = ctrl.Antecedent(np.arange(0, 20.1, 0.1), 'node_degree')
        self.distance_to_bs = ctrl.Antecedent(np.arange(0, 300.1, 0.1), 'distance_to_bs')
        self.link_quality = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'link_quality')
        
        # 鍒涘缓妯＄硦鍙橀噺 - 杈撳嚭
        self.cluster_head_chance = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'cluster_head_chance')
        self.next_hop_suitability = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'next_hop_suitability')
        
        # 瀹氫箟妯＄硦闆?- 鍓╀綑鑳介噺
        self.residual_energy['low'] = fuzz.trimf(self.residual_energy.universe, [0, 0, 0.4])
        self.residual_energy['medium'] = fuzz.trimf(self.residual_energy.universe, [0.2, 0.5, 0.8])
        self.residual_energy['high'] = fuzz.trimf(self.residual_energy.universe, [0.6, 1, 1])
        
        # 瀹氫箟妯＄硦闆?- 鑺傜偣涓績鎬?
        self.node_centrality['low'] = fuzz.trimf(self.node_centrality.universe, [0, 0, 0.4])
        self.node_centrality['medium'] = fuzz.trimf(self.node_centrality.universe, [0.2, 0.5, 0.8])
        self.node_centrality['high'] = fuzz.trimf(self.node_centrality.universe, [0.6, 1, 1])
        
        # 瀹氫箟妯＄硦闆?- 鑺傜偣搴︼Paperweight閭诲眳鏁伴噺锛?
        self.node_degree['few'] = fuzz.trimf(self.node_degree.universe, [0, 0, 6])
        self.node_degree['moderate'] = fuzz.trimf(self.node_degree.universe, [4, 8, 12])
        self.node_degree['many'] = fuzz.trimf(self.node_degree.universe, [10, 15, 20])
        
        # 瀹氫箟妯＄硦闆?- 鍒板熀绔欑殑璺濈
        self.distance_to_bs['close'] = fuzz.trimf(self.distance_to_bs.universe, [0, 0, 100])
        self.distance_to_bs['medium'] = fuzz.trimf(self.distance_to_bs.universe, [50, 150, 250])
        self.distance_to_bs['far'] = fuzz.trimf(self.distance_to_bs.universe, [200, 300, 300])
        
        # 瀹氫箟妯＄硦闆?- 閾捐矾璐ㄩ噺
        self.link_quality['poor'] = fuzz.trimf(self.link_quality.universe, [0, 0, 0.4])
        self.link_quality['average'] = fuzz.trimf(self.link_quality.universe, [0.3, 0.6, 0.8])
        self.link_quality['good'] = fuzz.trimf(self.link_quality.universe, [0.7, 1, 1])
        
        # 瀹氫箟妯＄硦闆?- 绨囧ご鏈轰細
        self.cluster_head_chance['very_low'] = fuzz.trimf(self.cluster_head_chance.universe, [0, 0, 0.25])
        self.cluster_head_chance['low'] = fuzz.trimf(self.cluster_head_chance.universe, [0.1, 0.3, 0.5])
        self.cluster_head_chance['medium'] = fuzz.trimf(self.cluster_head_chance.universe, [0.4, 0.6, 0.8])
        self.cluster_head_chance['high'] = fuzz.trimf(self.cluster_head_chance.universe, [0.7, 0.85, 1])
        self.cluster_head_chance['very_high'] = fuzz.trimf(self.cluster_head_chance.universe, [0.85, 1, 1])
        
        # 瀹氫箟妯＄硦闆?- 涓嬩竴璺抽€夋嫨锛堣兘
        self.next_hop_suitability['unsuitable'] = fuzz.trimf(self.next_hop_suitability.universe, [0, 0, 0.3])
        self.next_hop_suitability['less_suitable'] = fuzz.trimf(self.next_hop_suitability.universe, [0.2, 0.4, 0.6])
        self.next_hop_suitability['suitable'] = fuzz.trimf(self.next_hop_suitability.universe, [0.5, 0.7, 0.9])
        self.next_hop_suitability['very_suitable'] = fuzz.trimf(self.next_hop_suitability.universe, [0.8, 1, 1])
        
        # 瀹氫箟妯＄硦瑙勫垯 - 绨囧ご閫夋嫨瑙勫垯
        self.ch_rules = [
            # 瑙勫垯1锛氬鏋滆兘閲忛珮涓斾腑蹇冩€ч珮锛屽垯绨囧ご鏈轰細寰堥珮
            ctrl.Rule(self.residual_energy['high'] & self.node_centrality['high'], 
                      self.cluster_head_chance['very_high']),
            
            # 瑙勫垯2锛氬鏋滆兘閲忛珮涓斾腑蹇冩€т腑绛夛紝鍒欑皣澶存満浼氶珮
            ctrl.Rule(self.residual_energy['high'] & self.node_centrality['medium'], 
                      self.cluster_head_chance['high']),
            
            # 瑙勫垯3锛氬鏋滆兘閲忎腑绛変笖涓績鎬ч珮锛屽垯绨囧ご鏈轰細楂?
            ctrl.Rule(self.residual_energy['medium'] & self.node_centrality['high'], 
                      self.cluster_head_chance['high']),
            
            # 瑙勫垯4锛氬鏋滆兘閲忎腑绛変笖涓績鎬т腑绛夛紝鍒欑皣澶存満浼氫腑绛?
            ctrl.Rule(self.residual_energy['medium'] & self.node_centrality['medium'], 
                      self.cluster_head_chance['medium']),
            
            # 瑙勫垯5锛氬鏋滆兘閲忎綆锛屽垯绨囧ご鏈轰細寰堜綆
            ctrl.Rule(self.residual_energy['low'], self.cluster_head_chance['very_low']),
            
            # 瑙勫垯6锛氬鏋滀腑蹇冩€т綆锛屽垯绨囧ご鏈轰細浣?
            ctrl.Rule(self.node_centrality['low'], self.cluster_head_chance['low']),
            
            # 瑙勫垯7锛氬鏋滆兘閲忛珮涓旈偦灞呭锛屽垯绨囧ご鏈轰細寰堥珮
            ctrl.Rule(self.residual_energy['high'] & self.node_degree['many'], 
                      self.cluster_head_chance['very_high']),
            
            # 瑙勫垯8锛氬鏋滆兘閲忎腑绛変笖閭诲眳閫備腑锛屽垯绨囧ご鏈轰細涓瓑
            ctrl.Rule(self.residual_energy['medium'] & self.node_degree['moderate'], 
                      self.cluster_head_chance['medium']),
            
            # 瑙勫垯9锛氬鏋滃埌鍩虹珯璺濈杩戜笖鑳介噺楂橈紝鍒欑皣澶存満浼氶珮
            ctrl.Rule(self.distance_to_bs['close'] & self.residual_energy['high'], 
                      self.cluster_head_chance['high']),
            
            # 瑙勫垯10锛氬鏋滃埌鍩虹珯璺濈杩滀笖鑳介噺浣庯紝鍒欑皣澶存満浼氬緢浣?
            ctrl.Rule(self.distance_to_bs['far'] & self.residual_energy['low'],
                      self.cluster_head_chance['very_low']),

            # 鏂板瑙勫垯锛氱粨鍚堥摼璺川閲?
            # 瑙勫垯11: 濡傛灉閾捐矾璐ㄩ噺濂戒笖鑳介噺楂橈紝鍒欑皣澶存満浼氶潪甯搁珮
            ctrl.Rule(self.link_quality['good'] & self.residual_energy['high'],
                      self.cluster_head_chance['very_high']),
            
            # 瑙勫垯12: 濡傛灉閾捐矾璐ㄩ噺濂戒絾鑳介噺涓瓑锛屽垯绨囧ご鏈轰細楂?
            ctrl.Rule(self.link_quality['good'] & self.residual_energy['medium'],
                      self.cluster_head_chance['high']),

            # 瑙勫垯13: 濡傛灉閾捐矾璐ㄩ噺宸紝鍒欐樉钁楅檷浣庣皣澶存満浼?
            ctrl.Rule(self.link_quality['poor'], self.cluster_head_chance['very_low']),

            # 瑙勫垯14: 濡傛灉閾捐矾璐ㄩ噺濂斤紝涓旈偦灞呭锛屽垯鏄竴涓潪甯哥悊鎯崇殑绨囧ご
            ctrl.Rule(self.link_quality['good'] & self.node_degree['many'],
                      self.cluster_head_chance['very_high'])
        ]
        
        # 瀹氫箟妯＄硦瑙勫垯 - 涓嬩竴璺抽€夋嫨瑙勫垯
        self.nh_rules = [
            # 瑙勫垯1锛氬鏋滆兘閲忛珮涓旈摼璺川閲忓ソ涓旇窛绂诲熀绔欒繎锛屽垯闈炲父閫傚悎浣滀负涓嬩竴璺?
            ctrl.Rule(self.residual_energy['high'] & self.link_quality['good'] & 
                      self.distance_to_bs['close'], self.next_hop_suitability['very_suitable']),
            
            # 瑙勫垯2锛氬鏋滆兘閲忛珮涓旈摼璺川閲忓ソ锛屽垯閫傚悎浣滀负涓嬩竴璺?
            ctrl.Rule(self.residual_energy['high'] & self.link_quality['good'], 
                      self.next_hop_suitability['suitable']),
            
            # 瑙勫垯3锛氬鏋滆兘閲忎腑绛変笖閾捐矾璐ㄩ噺濂斤紝鍒欓€傚悎浣滀负涓嬩竴璺?
            ctrl.Rule(self.residual_energy['medium'] & self.link_quality['good'], 
                      self.next_hop_suitability['suitable']),
            
            # 瑙勫垯4锛氬鏋滆兘閲忎綆锛屽垯涓嶉€傚悎浣滀负涓嬩竴璺?
            ctrl.Rule(self.residual_energy['low'], self.next_hop_suitability['unsuitable']),
            
            # 瑙勫垯5锛氬鏋滈摼璺川閲忓樊锛屽垯涓嶅お閫傚悎浣滀负涓嬩竴璺?
            ctrl.Rule(self.link_quality['poor'], self.next_hop_suitability['less_suitable']),
            
            # 瑙勫垯6锛氬鏋滆兘閲忎腑绛変笖閾捐矾璐ㄩ噺涓€鑸紝鍒欎笉澶€傚悎浣滀负涓嬩竴璺?
            ctrl.Rule(self.residual_energy['medium'] & self.link_quality['average'], 
                      self.next_hop_suitability['less_suitable']),
            
            # 瑙勫垯7锛氬鏋滆窛绂诲熀绔欒繙涓旇兘閲忎綆锛屽垯涓嶉€傚悎浣滀负涓嬩竴璺?
            ctrl.Rule(self.distance_to_bs['far'] & self.residual_energy['low'], 
                      self.next_hop_suitability['unsuitable']),
            
            # 瑙勫垯8锛氬鏋滆窛绂诲熀绔欒繎涓旈摼璺川閲忓ソ锛屽垯闈炲父閫傚悎浣滀负涓嬩竴璺?
            ctrl.Rule(self.distance_to_bs['close'] & self.link_quality['good'], 
                      self.next_hop_suitability['very_suitable'])
        ]
        
        # 鍒涘缓鎺у埗绯荤粺
        self.ch_ctrl = ctrl.ControlSystem(self.ch_rules)
        self.nh_ctrl = ctrl.ControlSystem(self.nh_rules)
        
        # 鍒涘缓鎺у埗绯荤粺妯℃嫙鍣?
        self.ch_simulator = ctrl.ControlSystemSimulation(self.ch_ctrl)
        self.nh_simulator = ctrl.ControlSystemSimulation(self.nh_ctrl)
    
    def calculate_cluster_head_chance(self, residual_energy, node_centrality, node_degree, distance_to_bs, link_quality):
        """璁＄畻鑺傜偣鎴愪负绨囧ご鐨勬満浼?
        
        鍙傛暟:
            residual_energy (float): 鑺傜偣鍓╀綑鑳介噺姣斾緥 [0,1]
            node_centrality (float): 鑺傜偣涓績鎬?[0,1]
            node_degree (int): 鑺傜偣搴︼紙閭诲眳鏁伴噺锛?
            distance_to_bs (float): 鍒板熀绔欑殑璺濈
            link_quality (float): 閾捐矾璐ㄩ噺鎸囨暟 [0,1]
            
        杩斿洖:
            float: 鑺傜偣鎴愪负绨囧ご鐨勬満浼?[0,1]
        """
        # 璁剧疆杈撳叆鍊?
        self.ch_simulator.input['residual_energy'] = residual_energy
        self.ch_simulator.input['node_centrality'] = node_centrality
        self.ch_simulator.input['node_degree'] = min(node_degree, 20)  # 闄愬埗鍦ㄥ畾涔夎寖鍥村唴
        self.ch_simulator.input['distance_to_bs'] = min(distance_to_bs, 300)  # 闄愬埗鍦ㄥ畾涔夎寖鍥村唴
        self.ch_simulator.input['link_quality'] = link_quality
        
        # 璁＄畻
        try:
            self.ch_simulator.compute()
            return self.ch_simulator.output['cluster_head_chance']
        except:
            # 濡傛灉璁＄畻澶辫触锛屼娇鐢ㄥ姞鏉冨钩鍧囧€间綔涓哄閫夋柟娉?
            weighted_sum = (residual_energy * 0.4 + 
                           node_centrality * 0.3 + 
                           min(node_degree / 20, 1) * 0.2 + 
                           (1 - min(distance_to_bs / 300, 1)) * 0.1)
            return weighted_sum
    
    def calculate_next_hop_suitability(self, residual_energy, link_quality, distance_to_bs):
        """璁＄畻鑺傜偣浣滀负涓嬩竴璺崇殑閫傚悎搴?
        
        鍙傛暟:
            residual_energy (float): 鑺傜偣鍓╀綑鑳介噺姣斾緥 [0,1]
            link_quality (float): 閾捐矾璐ㄩ噺 [0,1]
            distance_to_bs (float): 鍒板熀绔欑殑璺濈
            
        杩斿洖:
            float: 鑺傜偣浣滀负涓嬩竴璺崇殑閫傚悎搴?[0,1]
        """
        # 璁剧疆杈撳叆鍊?
        self.nh_simulator.input['residual_energy'] = residual_energy
        self.nh_simulator.input['link_quality'] = link_quality
        self.nh_simulator.input['distance_to_bs'] = min(distance_to_bs, 300)  # 闄愬埗鍦ㄥ畾涔夎寖鍥村唴
        
        # 璁＄畻
        try:
            self.nh_simulator.compute()
            return self.nh_simulator.output['next_hop_suitability']
        except:
            # 濡傛灉璁＄畻澶辫触锛屼娇鐢ㄥ姞鏉冨钩鍧囧€间綔涓哄閫夋柟娉?
            weighted_sum = (residual_energy * 0.4 + 
                           link_quality * 0.4 + 
                           (1 - min(distance_to_bs / 300, 1)) * 0.2)
            return weighted_sum

    def visualize_membership_functions(self):
        """鍙鍖栨ā绯婇泦鐨勯毝灞炲害鍑芥暟"""
        import matplotlib.pyplot as plt
        
        # 璁剧疆椋庢牸鍜屽瓧浣?
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.titlesize': 18
        })
        
        # 鍒涘缓涓€涓ぇ鐨勫浘褰紝鍖呭惈鎵€鏈夐毝灞炲害鍑芥暟
        fig = plt.figure(figsize=(15, 18), dpi=100)
        fig.suptitle('Fuzzy Logic Membership Functions', fontsize=22, fontweight='bold', y=0.98)
        fig.set_facecolor('#f8f9fa')
        
        # 璁剧疆棰滆壊鏂规
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
        
        # 鍒涘缓瀛愬浘甯冨眬
        gs = fig.add_gridspec(4, 2, hspace=0.4, wspace=0.3)
        
        # 鑷畾涔夊彲瑙嗗寲杈撳叆鍙橀噺 - 鍓╀綑鑳介噺
        ax1 = fig.add_subplot(gs[0, 0])
        x = np.arange(0, 1.01, 0.01)
        for i, term in enumerate(['low', 'medium', 'high']):
            ax1.plot(x, self.residual_energy[term].mf, 
                    linewidth=2.5, label=term.capitalize(), color=colors[i])
        ax1.set_title('Residual Energy', fontweight='bold', pad=10)
        ax1.set_xlabel('Energy Ratio [0-1]')
        ax1.set_ylabel('Membership Degree')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend(loc='best')
        ax1.set_ylim([0, 1.1])
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # 鑷畾涔夊彲瑙嗗寲杈撳叆鍙橀噺 - 鑺傜偣涓績鎬?
        ax2 = fig.add_subplot(gs[0, 1])
        for i, term in enumerate(['low', 'medium', 'high']):
            ax2.plot(x, self.node_centrality[term].mf, 
                    linewidth=2.5, label=term.capitalize(), color=colors[i])
        ax2.set_title('Node Centrality', fontweight='bold', pad=10)
        ax2.set_xlabel('Centrality Value [0-1]')
        ax2.set_ylabel('Membership Degree')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend(loc='best')
        ax2.set_ylim([0, 1.1])
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        # 鑷畾涔夊彲瑙嗗寲杈撳叆鍙橀噺 - 鑺傜偣搴?
        ax3 = fig.add_subplot(gs[1, 0])
        x_degree = np.arange(0, 20.1, 0.1)
        for i, term in enumerate(['few', 'moderate', 'many']):
            ax3.plot(x_degree, self.node_degree[term].mf, 
                    linewidth=2.5, label=term.capitalize(), color=colors[i])
        ax3.set_title('Node Degree', fontweight='bold', pad=10)
        ax3.set_xlabel('Number of Neighbors')
        ax3.set_ylabel('Membership Degree')
        ax3.grid(True, linestyle='--', alpha=0.7)
        ax3.legend(loc='best')
        ax3.set_ylim([0, 1.1])
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        
        # 鑷畾涔夊彲瑙嗗寲杈撳叆鍙橀噺 - 鍒板熀绔欑殑璺濈
        ax4 = fig.add_subplot(gs[1, 1])
        x_dist = np.arange(0, 300.1, 0.1)
        for i, term in enumerate(['close', 'medium', 'far']):
            ax4.plot(x_dist, self.distance_to_bs[term].mf, 
                    linewidth=2.5, label=term.capitalize(), color=colors[i])
        ax4.set_title('Distance to Base Station', fontweight='bold', pad=10)
        ax4.set_xlabel('Distance (m)')
        ax4.set_ylabel('Membership Degree')
        ax4.grid(True, linestyle='--', alpha=0.7)
        ax4.legend(loc='best')
        ax4.set_ylim([0, 1.1])
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        
        # 鑷畾涔夊彲瑙嗗寲杈撳叆鍙橀噺 - 閾捐矾璐ㄩ噺
        ax5 = fig.add_subplot(gs[2, 0])
        for i, term in enumerate(['poor', 'average', 'good']):
            ax5.plot(x, self.link_quality[term].mf, 
                    linewidth=2.5, label=term.capitalize(), color=colors[i])
        ax5.set_title('Link Quality', fontweight='bold', pad=10)
        ax5.set_xlabel('Quality Value [0-1]')
        ax5.set_ylabel('Membership Degree')
        ax5.grid(True, linestyle='--', alpha=0.7)
        ax5.legend(loc='best')
        ax5.set_ylim([0, 1.1])
        ax5.spines['top'].set_visible(False)
        ax5.spines['right'].set_visible(False)
        
        # 鑷畾涔夊彲瑙嗗寲杈撳嚭鍙橀噺 - 绨囧ご鏈轰細
        ax6 = fig.add_subplot(gs[2, 1])
        for i, term in enumerate(['very_low', 'low', 'medium', 'high', 'very_high']):
            ax6.plot(x, self.cluster_head_chance[term].mf, 
                    linewidth=2.5, label=' '.join(term.split('_')).capitalize(), color=colors[i % len(colors)])
        ax6.set_title('Cluster Head Chance', fontweight='bold', pad=10)
        ax6.set_xlabel('Chance Value [0-1]')
        ax6.set_ylabel('Membership Degree')
        ax6.grid(True, linestyle='--', alpha=0.7)
        ax6.legend(loc='best')
        ax6.set_ylim([0, 1.1])
        ax6.spines['top'].set_visible(False)
        ax6.spines['right'].set_visible(False)
        
        # 鑷畾涔夊彲瑙嗗寲杈撳嚭鍙橀噺 - 涓嬩竴璺抽€夋嫨
        ax7 = fig.add_subplot(gs[3, 0])
        for i, term in enumerate(['unsuitable', 'less_suitable', 'suitable', 'very_suitable']):
            ax7.plot(x, self.next_hop_suitability[term].mf, 
                    linewidth=2.5, label=' '.join(term.split('_')).capitalize(), color=colors[i % len(colors)])
        ax7.set_title('Next Hop Suitability', fontweight='bold', pad=10)
        ax7.set_xlabel('Suitability Value [0-1]')
        ax7.set_ylabel('Membership Degree')
        ax7.grid(True, linestyle='--', alpha=0.7)
        ax7.legend(loc='best')
        ax7.set_ylim([0, 1.1])
        ax7.spines['top'].set_visible(False)
        ax7.spines['right'].set_visible(False)
        
        # 娣诲姞璇存槑鏂囨湰
        description = (
            "Fuzzy Logic System for WSN Decision Making\n"
            "This visualization shows the membership functions used in the fuzzy inference system.\n"
            "These functions help translate crisp input values into fuzzy linguistic variables for decision making."
        )
        fig.text(0.5, 0.02, description, ha='center', fontsize=12, style='italic', 
                bbox=dict(facecolor='#f8f9fa', alpha=0.8, boxstyle='round,pad=0.5', edgecolor='#d5dbdb'))
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.94, bottom=0.08)
        plt.savefig('fuzzy_membership_functions.svg', dpi=300, bbox_inches='tight')
        plt.show()

    def visualize_fuzzy_surface(self):
        """鍙鍖栨ā绯婃帶鍒惰〃闈?""
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        
        # 璁剧疆椋庢牸鍜屽瓧浣?
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 12,
            'figure.titlesize': 18
        })
        
        # 鍒涘缓涓€涓ぇ鐨勫浘褰紝鍖呭惈鎵€鏈夋帶鍒惰〃闈?
        fig = plt.figure(figsize=(16, 10), dpi=100)
        fig.suptitle('Fuzzy Logic Control Surfaces', fontsize=22, fontweight='bold', y=0.98)
        fig.set_facecolor('#f8f9fa')
        
        # 鍒涘缓鑷畾涔夌殑3D琛ㄩ潰鍥?- 绨囧ご閫夋嫨锛堣兘閲弙s涓績鎬э級
        ax1 = fig.add_subplot(121, projection='3d')
        x_energy = np.arange(0, 1.01, 0.01)
        y_centrality = np.arange(0, 1.01, 0.01)
        X, Y = np.meshgrid(x_energy, y_centrality)
        Z = np.zeros_like(X)
        
        # 璁＄畻姣忎釜鐐圭殑杈撳嚭鍊?
        for i in range(len(x_energy)):
            for j in range(len(y_centrality)):
                self.ch_simulator.input['residual_energy'] = x_energy[i]
                self.ch_simulator.input['node_centrality'] = y_centrality[j]
                self.ch_simulator.input['node_degree'] = 10  # 鍥哄畾鍊?
                self.ch_simulator.input['distance_to_bs'] = 150  # 鍥哄畾鍊?
                try:
                    self.ch_simulator.compute()
                    Z[j, i] = self.ch_simulator.output['cluster_head_chance']
                except:
                    Z[j, i] = 0.5  # 榛樿鍊?
        
        # 缁樺埗3D琛ㄩ潰
        surf1 = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, 
                               linewidth=0, antialiased=True, edgecolor='none')
        
        # 娣诲姞棰滆壊鏉″拰鏍囩
        cbar1 = fig.colorbar(surf1, ax=ax1, shrink=0.6, pad=0.1)
        cbar1.set_label('Cluster Head Chance')
        
        ax1.set_title('Cluster Head Selection Surface', fontweight='bold', pad=10)
        ax1.set_xlabel('Residual Energy')
        ax1.set_ylabel('Node Centrality')
        ax1.set_zlabel('CH Chance')
        ax1.view_init(elev=30, azim=45)  # 璁剧疆瑙嗚
        
        # 鍒涘缓鑷畾涔夌殑3D琛ㄩ潰鍥?- 涓嬩竴璺抽€夋嫨锛堣兘閲弙s閾捐矾璐ㄩ噺锛?
        ax2 = fig.add_subplot(122, projection='3d')
        x_energy = np.arange(0, 1.01, 0.01)
        y_link = np.arange(0, 1.01, 0.01)
        X, Y = np.meshgrid(x_energy, y_link)
        Z = np.zeros_like(X)
        
        # 璁＄畻姣忎釜鐐圭殑杈撳嚭鍊?
        for i in range(len(x_energy)):
            for j in range(len(y_link)):
                self.nh_simulator.input['residual_energy'] = x_energy[i]
                self.nh_simulator.input['link_quality'] = y_link[j]
                self.nh_simulator.input['distance_to_bs'] = 150  # 鍥哄畾鍊?
                try:
                    self.nh_simulator.compute()
                    Z[j, i] = self.nh_simulator.output['next_hop_suitability']
                except:
                    Z[j, i] = 0.5  # 榛樿鍊?
        
        # 缁樺埗3D琛ㄩ潰
        surf2 = ax2.plot_surface(X, Y, Z, cmap='plasma', alpha=0.8, 
                               linewidth=0, antialiased=True, edgecolor='none')
        
        # 娣诲姞棰滆壊鏉″拰鏍囩
        cbar2 = fig.colorbar(surf2, ax=ax2, shrink=0.6, pad=0.1)
        cbar2.set_label('Next Hop Suitability')
        
        ax2.set_title('Next Hop Selection Surface', fontweight='bold', pad=10)
        ax2.set_xlabel('Residual Energy')
        ax2.set_ylabel('Link Quality')
        ax2.set_zlabel('NH Suitability')
        ax2.view_init(elev=30, azim=45)  # 璁剧疆瑙嗚
        
        # 娣诲姞璇存槑鏂囨湰
        description = (
            "Fuzzy Logic Control Surfaces for WSN Decision Making\n"
            "This visualization shows the membership functions used in the fuzzy inference system.\n"
            "These functions help translate crisp input values into fuzzy linguistic variables for decision making."
        )
        fig.text(0.5, 0.02, description, ha='center', fontsize=12, style='italic', 
                bbox=dict(facecolor='#f8f9fa', alpha=0.8, boxstyle='round,pad=0.5', edgecolor='#d5dbdb'))
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92, bottom=0.1)
        plt.savefig('fuzzy_control_surfaces.svg', dpi=300, bbox_inches='tight')
        plt.show()

# 娴嬭瘯浠ｇ爜
if __name__ == "__main__":
    # 鍒涘缓妯＄硦閫昏緫绯荤粺
    fls = FuzzyLogicSystem()
    
    # 娴嬭瘯绨囧ご閫夋嫨
    ch_chance = fls.calculate_cluster_head_chance(
        residual_energy=0.8,  # 楂樺墿浣欒兘閲?
        node_centrality=0.7,  # 楂樹腑蹇冩€?
        node_degree=12,       # 閫備腑鐨勯偦灞呮暟閲?
        distance_to_bs=80     # 杈冭繎鐨勫熀绔欒窛绂?
    )
    print(f"[FUZZY] CH selection likelihood: {ch_chance:.4f}")
    
    # 娴嬭瘯涓嬩竴璺抽€夋嫨
    nh_suitability = fls.calculate_next_hop_suitability(
        residual_energy=0.6,  # 涓瓑鍓╀綑鑳介噺
        link_quality=0.8,     # 鑹ソ鐨勯摼璺川閲?
        distance_to_bs=120    # 涓瓑鐨勫熀绔欒窛绂?
    )
    print(f"[FUZZY] Next-hop suitability: {nh_suitability:.4f}")
    
    # 鍙鍖?
    # fls.visualize_membership_functions()
    # fls.visualize_fuzzy_surface()

