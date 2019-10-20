#
#
# import itertools
# import collections
# from sklearn.cluster import KMeans
# import numpy as np
# import bisect
# X = np.array([[1, 2], [1, 4], [1, 0],
#               [10, 2], [10, 4], [10, 0]])
# kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
#
#
#
# import math
#
# def k_means():
#     dataset = []
#     with open("input") as file:
#         for line in file:
#
#             dataset.append(line.strip("\n"))
#     K = int(dataset[0].split()[1])
#     N = int(dataset[0].split()[0])
#     input = []
#     for data in dataset[1:1 + N]:
#         coords = data.split()
#         input.append(tuple([float(coords[i]) for i in range(len(coords))]))
#     centers = []
#     for data in dataset[1 + N:]:
#         coords = data.split()
#         centers.append(tuple([float(coords[i]) for i in range(len(coords))]))
#     clusters = [set() for i in range(K)]
#
#     dimension = len(input[1])
#
#     # initialization
#     while True:
#         new_clusters = [set() for i in range(K)]
#         for point in input:
#             min_dist = 99999999
#             min_center = -1
#             for i in range(len(centers)):
#                 distance = 0
#                 for d in range(dimension):
#                     distance += (point[d] - centers[i][d]) ** 2
#                 distance = math.sqrt(distance)
#                 if distance < min_dist:
#                     min_dist = distance
#                     min_center = i
#             new_clusters[min_center].add(point)
#         end = True
#         for i in range(len(new_clusters)):
#             if not new_clusters[i].issubset(clusters[i]) or not clusters[i].issubset(new_clusters[i]):
#                 end = False
#                 break
#         # if end:
#         #    clusters = new_clusters
#         #    break
#         first_check = end
#         new_assignment = [set() for i in range(K)]
#         new_centers = []
#         clusters = new_clusters
#         for cluster_index in range(len(clusters)):
#             avg_attributes = [0 for i in range(dimension)]
#             num_points = len(clusters[cluster_index])
#             for item in clusters[cluster_index]:
#                 for d in range(dimension):
#                     avg_attributes[d] += item[d]
#             for attr_index in range(len(avg_attributes)):
#                 avg_attributes[attr_index] /= num_points
#             new_centers.append(avg_attributes)
#         end = True
#         # for i in range(len(centers)):
#         #    if (abs(centers[i][0] - new_centers[i][0]) > 0.00001 and abs(centers[i][1] - new_centers[i][1]) > 0.00001):
#         #        end = False
#         #        break
#         if end and first_check:
#             break
#         centers = new_centers
#         break
#     output = []
#     for point in input:
#         for i in range(len(clusters)):
#             if point in clusters[i]:
#                 output.append(i)
#                 break
#     for i in output:
#         print(i)
#     print(new_centers)
#
# def find_dist(cluster1,cluster2,dimension):
#     min_dist = 99999999
#     smallest_point = None
#     for (pt1,pt2)  in itertools.product(cluster1,cluster2):
#         dist = 0
#         for d in range(dimension):
#             dist += (pt1[d] - pt2[d])**2
#         dist = math.sqrt(dist)
#         if min_dist > dist:
#             min_dist = dist
#     return min_dist
#
#
# def agnes():
#     dataset = []
#     with open("input") as file:
#         for line in file:
#             dataset.append(line.strip("\n"))
#     K = int(dataset[0].split()[1])
#     N = int(dataset[0].split()[0])
#     input = []
#     for data in dataset[1:1 + N]:
#         coords = data.split()
#         input.append(tuple([float(coords[i]) for i in range(len(coords))]))
#     centers = []
#     point_to_dic = {}
#     pt_id = 0
#     for point in input:
#         point_to_dic[point] = pt_id
#         pt_id += 1
#     for data in dataset[1 + N:]:
#         coords = data.split()
#         centers.append(tuple([float(coords[i]) for i in range(len(coords))]))
#
#     dimension = len(input[1])
#
#     clusters = [{i} for i in input]
#
#
#     counter = len(clusters)
#
#     id_to_cluster = {}
#     for i in range(len(clusters)):
#         id_to_cluster[i] = clusters[i]
#     D_mat = collections.defaultdict(lambda : collections.defaultdict(float))
#     for (i,j) in itertools.product(list(id_to_cluster.keys()),repeat=2):
#         if i == j:
#             continue
#         dis = find_dist(id_to_cluster[i],id_to_cluster[j],dimension)
#         D_mat[i][j] = dis
#
#     while True:
#         min_dist = 99999999
#         min_pair = (-1,-1)
#         all_id = list(id_to_cluster.keys())
#         all_id.sort()
#         for (i,j) in itertools.product(all_id,repeat=2):
#             if i == j :
#                 continue
#             dist = D_mat[i][j]
#             if dist < min_dist:
#                 min_dist = dist
#                 min_pair = (i, j)
#         id_to_cluster[min_pair[0]].update(id_to_cluster[min_pair[1]])
#         id_to_cluster.pop(min_pair[1])
#         D_mat[min_pair[0]].pop(min_pair[1])
#         if len(D_mat[min_pair[0]]) == 0:
#             D_mat.pop(min_pair[0])
#         #update d
#         for cid,cluster in id_to_cluster.items():
#             if cid == min_pair[0]:
#                 continue#update from the updated to
#             else:
#                 dist = find_dist(cluster,id_to_cluster[min_pair[0]],dimension)
#                 D_mat[cid][min_pair[0]] = dist
#                 D_mat[min_pair[0]][cid] = dist
#         counter -= 1
#         if counter == K:
#             break
#
#     output = []
#
#     for pt in input:
#         for cid, cluster in id_to_cluster.items():
#             if pt in cluster:
#                 output.append(cid)
#                 break
#     for cid in output:
#         print(cid)
import queue
from itertools import permutations
from time import time
from collections import Counter

class Solution:
    def __init__(self):
        self.ans = []

    def findSubsequences(self, nums):
        import copy

        if len(nums) < 2:
            return self.ans
        dic = {}
        for i in range(len(nums)):
            if nums[i] not in dic:
                self.dfs(nums, i, [])
                dic[nums[i]] = 1
        return self.ans

    def dfs(self, nums, index, prefix):
        if index >= len(nums):
            return
        appended_item = False
        if len(prefix) == 0 or nums[index] >= prefix[-1]:
            prefix.append(nums[index])
            appended_item = True
            if len(prefix) > 1:
                self.ans.append(copy.deepcopy(prefix))

        i = index + 1
        dic = {}
        while i < len(nums):
            if nums[i] not in dic:
                dic[nums[i]] = 1
                self.dfs(nums, i, prefix)
            i += 1
        if appended_item:
            prefix.pop(len(prefix) - 1)

    def carFleet(self, target, position, speed) -> int:
        comb = [[position[i],speed[i]]for i in range(len(position))]

        comb.sort(key=lambda element:element[0],reverse=True)
        counter = 0
        head = comb[0]
        for i in range(len(comb)):
            if comb[i][0] == head[0]:
                continue
            else:
                if (target - head[0])/head[1] < (target - comb[i][0])/comb[i][1]:
                    head = comb[i]
                    counter += 1
        print(counter)
        return counter

    def findClosestElements(self, arr, k: int, x: int):

        i = bisect.bisect_left(arr,x)
        if arr[i] == x:
            total_num = 2 * k - 1
            start = max(0, i - k + 1)
        else:
            total_num = 2 * k
            start = max(0, i - k)


        end = min(total_num + start,len(arr))
        new_arr = [0] + arr[start : end]
        #new_arr[0] = abs(new_arr[0] - x)
        for i in range(1,len(new_arr)):
            new_arr[i] = abs(new_arr[i] - x) + new_arr[i - 1]
        min_val = 999999999
        index = -1
        for i in range(len(new_arr) - k ):
            if min_val > new_arr[i + k] - new_arr[i]:
                min_val = new_arr[i + k] - new_arr[i]
                index = i
        return arr[start + index:start + index + k]

    def reorganizeString(self, S: str) -> str:
        if len(S) < 2:
            return S
        pool = [[0, chr(i)] for i in range(97, 97 + 26)]
        for i in range(len(S)):
            pool[ord(S[i]) - 97][0] -= 1
        ans = []
        new_pool = []
        for item in pool:
            if item[0] != 0:
                new_pool.append(item)
        pool = new_pool
        import heapq
        heapq.heapify(pool)
        ans = []
        while len(pool) > 0:
            maximum = heapq.heappop(pool)
            if len(ans) != 0 and ans[-1] == maximum[1]:
                if len(pool) == 0:
                    return ""
                else:
                    second_maximum = heapq.heappop(pool)
                    heapq.heappush(pool,maximum)
                    maximum = second_maximum
            maximum[0] -= 1
            ans.append(maximum[1])
            if maximum[0] != 0:
                heapq.heappush(pool,maximum)
        return "".join(ans)

    def matrixScore(self, A) -> int:
        def converter(digits):
            summation = 0
            for i in range(len(digits)):
                summation += digits[len(digits) - 1 - i] * pow(2, i)
            return summation

        def calculator(matrix):
            summation = 0
            for num in matrix:
                summation += converter(num)
            return summation

        def flip_row(row):
            for i in range(len(row)):
                row[i] ^= 1

        import copy
        first = copy.deepcopy(A)
        second = copy.deepcopy(A)
        for num in first:
            num[0] ^= 1
        for num in first:
            if num[0] == 0:
                flip_row(num)
        first_sum = calculator(first)
        for num in second:
            if num[0] == 0:
                flip_row(num)
        second_sum = calculator(second)
        A = second if second_sum > first_sum else first
        for i in range(1, len(A[0])):
            counter = 0
            for row in A:
                counter += row[i]
            if counter <= len(A) // 2:
                for row in A:
                    row[i] ^= 1
        return calculator(A)

    def videoStitching(self, clips, T):

        dp_count = [0] * (T + 1)
        import copy

        clips.sort(key=lambda element: [element[0],element[1] - element[0]])



        end_pt = 0
        ans = 0

        furtherst = -1
        while furtherst < T:
            original_furtherst = furtherst
            for j in range(0,len(clips)):
                if clips[j][0] > end_pt:
                    break
                furtherst = max(furtherst,clips[j][1])
            if furtherst == end_pt:
                return -1
            end_pt = furtherst
            ans += 1




        return ans
    def videoStitching1(self, clips, T):

        dp_count = [0] * (T + 1)
        import copy

        clips.sort(key=lambda element: element[0])
        dp = [copy.deepcopy(clips)]


        for i in range(T):
            i = i + 1
            cur_copy = None
            cur_cut = 999999999
            for j in range(0, i):
                index = -1
                minimum = 999
                for k in range(len(dp[j])):
                    clip = dp[j][k]
                    if clip[0] > j or clip[1] < i:
                        continue
                    if clip[0] == j and clip[1] == i:
                        index = k
                        minimum = 0
                        break
                    else:
                        if minimum > total_cut_for_this_clip:
                            minimum = total_cut_for_this_clip
                            index = k
                if minimum == 999:
                    continue
                tmp = copy.deepcopy(dp[j])
                tmp.pop(index)
                steps = minimum + dp_count[j]

                if steps < cur_cut:
                    cur_cut = steps
                    cur_copy = tmp
            if cur_copy is None:
                return -1
            dp_count[i] = cur_cut
            dp.append(cur_copy)
        return dp_count[-1]

    def mincostTickets(self, days, costs) :
        dp_cost = [0] * (len(days) + 1)
        dp_coverage = [0] * (len(days) + 1)
        new_costs = [[costs[0],0],[costs[1],6],[costs[2],29]]
        new_costs.sort(key=lambda element:element[0])
        dp_cost[1] = new_costs[0][0]
        dp_coverage[1] = new_costs[0][1]
        days = [0] + days
        for i in range(2, len(days)):
            choices = []

            if dp_coverage[i - 1] < days[i] - days[i - 1]:
                choices.append([costs[0] + dp_cost[i - 1],0])
            else:
                choices.append([dp_cost[i - 1],0])
            for j in range(i - 1,-1,-1):
                if days[i] - days[j] >= 7:
                    break

            last_covered_day = dp_coverage[j] + days[j]
            still_covered = True
            for k in range(j,i + 1):
                if days[k]  > last_covered_day:
                    still_covered = False
                    break
            if not still_covered:
                choices.append([dp_cost[k - 1] + costs[1],6 - (days[i] - days[k])])
            else:
                choices.append([dp_cost[k - 1],last_covered_day - days[i] ])



            for j in range(i - 1,-1,-1):
                if days[i] - days[j] >= 30:
                    break

            last_covered_day = dp_coverage[j] + days[j]
            for k in range(j,i + 1):
                if days[k]  > last_covered_day:
                    break
            still_covered = True
            for k in range(j + 1,i + 1):
                if days[k] > last_covered_day:
                    still_covered = False
                    break
            if not still_covered:
                choices.append([dp_cost[k - 1] + costs[2],29 - (days[i] - days[k])])
            else:
                choices.append([dp_cost[k - 1] ,last_covered_day - days[i] ])
            choices.sort(key = lambda element:element[0])
            dp_cost[i] = choices[0][0]
            dp_coverage[i] = choices[0][1]
        return dp_cost[-1]

    def findNumberOfLIS(self, nums):
        dp = [0] * (len(nums))
        dp_num = [0] * (len(nums))
        if len(nums) < 2:
            return len(nums)
        dp[0] = 1
        dp_num[0] = 1
        cur_longest = 1
        cur_num = 1

        for i in range(1,len(nums)):
            tmp_longest = 1
            tmp_longest_num = 0

            for j in range(0, i):

                if nums[i] > nums[j] and dp[j] + 1 > tmp_longest:
                    tmp_longest = dp[j] + 1
                    tmp_longest_num = dp_num[j]
                elif nums[i] > nums[j] and dp[j] + 1 == tmp_longest:
                    tmp_longest_num +=  dp_num[j]
            if tmp_longest_num == 0:
                tmp_longest_num += 1
            if tmp_longest > cur_longest:
                cur_longest = tmp_longest
                cur_num = tmp_longest_num

            elif tmp_longest == cur_longest:
                cur_num += tmp_longest_num
            # print(tmp_longest)
            dp[i] = tmp_longest
            dp_num[i] = tmp_longest_num


        return cur_num

    def largestSumOfAverages(self, A,K: int) -> float:
        cumsum = []
        prev = 0.0
        for i in A:
            prev += i
            cumsum.append(prev)
        dp = [[0] * len(A) for i in range(K)]
        for i in range(len(dp[0])):

            dp[0][i] = (cumsum[i])/(i + 1)
        for i in range(1, K):
            cc = 3
            for j in range(len(A)):
                if j < i:
                    continue
                maximum = 0
                for kk in range(j):
                    if kk < i - 1:
                        continue
                    #print((cumsum[j + 1] - cumsum[kk]) / (j - kk) + dp[i - 1][kk])
                    dp[i][j] = max((cumsum[j] - cumsum[kk]) / (j  - kk) + dp[i - 1][kk], dp[i][j])
        print(dp)
        return dp[-1][-1]

    def countSubstrings(self, s: str) -> int:
        dp = [[False] * len(s) for i in range(len(s))]
        total = 0
        for i in range(len(s)):
            for j in range(0, i + 1):
                if i == j:
                    dp[i][j] = True
                else:
                    if i - 1 == j:
                        dp[i][j] = s[i] == s[j]
                    else:
                        dp[i][j] = dp[j + 1][i - 1] and s[i] == s[j]
                if dp[i][j]:
                    total += 1
        for r in dp:
            print(r)
        return total

    def subarrayBitwiseORs(self, A):
        ans = set()
        cur = {0}
        for x in A:
            cur = {x | y for y in cur} | {x}
            ans |= cur
        return len(ans)

    def maxProfit(self, prices, fee) :
        if len(prices) < 2:
            return 0
        ans = 0
        max_profit = 0
        holding_price = 99999
        ptr = 0
        while ptr < len(prices):
            if max_profit == 0 and holding_price > prices[ptr]:
                holding_price = prices[ptr]
            elif max_profit == 0 and prices[ptr] > holding_price + 2:
                sell_price = prices[ptr]

                max_profit += (sell_price - holding_price)
                holding_price = prices[ptr]
            elif max_profit != 0 and prices[ptr] > holding_price:
                sell_price = prices[ptr]

                max_profit += (sell_price - holding_price)
                holding_price = prices[ptr]
            elif holding_price - fee > prices[ptr]:
                ans += max_profit - fee
                max_profit = 0
                holding_price = prices[ptr]
            ptr += 1

        return ans + max(max_profit - 2,0)


    def maxA(self, N: int) -> int:
        dp = list(range(N + 1))
        for n in range(7, N + 1):
            dp[n] = max(dp[n - 1] + 1, dp[n - 3] * 2, dp[n - 4] * 3, dp[n - 5] * 4)
        return dp[-1]
    def fake(self,N):
        dp = list(range(N + 1))
        for n in range(7, N + 1):
            dp[n] = max(dp[n - 1] + 1, dp[n - 3] * 2, dp[n - 4] * 3, dp[n - 5] * 4,dp[n - 6] * 5)
        return dp[-1]
    def new21Game(self, N: int, K: int, W: int) -> float:
        dp = [0 for i in range(N)]
        dp[0] = 1

    def hitBricks(self, grid,hits) :
        if len(grid) == 0 or len(grid[0]) == 0:
            return [0 for i in range(len(hits))]

        connectivity = [[[0, 0, 0, 0] for j in range(len(grid[0]))] for i in range(len(grid))]

        def link(i, j, grid, direc):
            if i >= len(grid) or i < 0 or j >= len(grid[0]) or j < 0 or grid[i][j] != 1:
                return 0
            connectivity[i][j][direc] = 1
            grid[i][j] = -1
            link(i - 1, j, grid, 2)
            link(i + 1, j, grid, 0)
            link(i, j - 1, grid, 1)
            link(i, j + 1, grid, 3)
            grid[i][j] = 1

        def unlink(i, j, grid, direc, p):
            if i >= len(grid) or i < 0 or j >= len(grid[0]) or j < 0 or grid[i][j] == 0 or i == 0 or connectivity[i][j][
                direc] == 0:
                return 0
            connectivity[i][j][direc] = 0
            if sum(connectivity[i][j]) == 0:
                summation = 1
                grid[i][j] = 0
            else:
                summation = 0

            if p != (i - 1, j) and (
                    (connectivity[i][j][0] == 1 and sum(connectivity[i][j]) == 1) or sum(connectivity[i][j]) == 0):
                summation += unlink(i - 1, j, grid, 2, (i, j))
            if p != (i + 1, j) and (
                    (connectivity[i][j][2] == 1 and sum(connectivity[i][j]) == 1) or sum(connectivity[i][j]) == 0):
                summation += unlink(i + 1, j, grid, 0, (i, j))
            if p != (i, j - 1) and (
                    (connectivity[i][j][3] == 1 and sum(connectivity[i][j]) == 1) or sum(connectivity[i][j]) == 0):
                summation += unlink(i, j - 1, grid, 1, (i, j))
            if p != (i, j + 1) and (
                    (connectivity[i][j][1] == 1 and sum(connectivity[i][j]) == 1) or sum(connectivity[i][j]) == 0):
                summation += unlink(i, j + 1, grid, 3, (i, j))
            return summation

        ans = []

        for i in range(len(grid[0])):
            link(0, i, grid, 0)

        for hit in hits:

            i, j = hit
            grid[i][j] = 0
            print("##########3")
            for r in grid:
                print(r)
            print("##########3")
            connectivity[i][j] = [0, 0, 0, 0]
            up = unlink(i - 1, j, grid, 2, hit)
            down = unlink(i + 1, j, grid, 0, hit)
            left = unlink(i, j - 1, grid, 1, hit)
            right = unlink(i, j + 1, grid, 3, hit)
            ans.append(up + down + left + right)
        return ans
# class Foo:
#     def __init__(self):
#         print("third")
#         self.num = 0
#         import threading
#         self.mutex = threading.Lock()
#         self.cv1 = threading.Condition(self.mutex)
#         self.cv2 = threading.Condition(self.mutex)
#         self.cv3 = threading.Condition(self.mutex)
#
#     def first(self, printFirst: 'Callable[[], None]') -> None:
#         print("first")
#         # printFirst() outputs "first". Do not change or remove this line.
#         self.mutex.acquire()
#         printFirst()
#         self.num += 1
#         self.cv2.notify()
#         self.mutex.release()
#         print("first")
#
#     def second(self, printSecond: 'Callable[[], None]') -> None:
#         print("second")
#         # printSecond() outputs "second". Do not change or remove this line.
#         self.mutex.acquire()
#         while self.num != 1:
#             self.cv2.wait()
#         self.num += 1
#         printSecond()
#         self.cv3.notify()
#         self.mutex.release()
#         print("third")
#
#     def third(self, printThird: 'Callable[[], None]') -> None:
#
#         # printThird() outputs "third". Do not change or remove this line.
#         self.mutex.acquire()
#         while self.num != 2:
#             self.cv3.wait()
#         self.num += 1
#         printThird()
#         self.mutex.release()

    def shortestSubarray(self, A, K):
        prefix = [0] * len(A)
        total = 0
        for i in range(len(A)):
            prefix[i] = total
            total += A[i]
        end, st = 0, 0
        total = 0
        ans = len(A) + 1
        while end < len(A):
            if A[end] < 0:
                if total + A[end] < 1:
                    st = end + 1

                    total = 0
                else:
                    total += A[end]
            else:
                total += A[end]
                cur = A[end]
                if total >= K:

                    new_st = end
                    while new_st > st and total - prefix[new_st] + prefix[st] < K:
                        new_st -= 1

                    total = total - prefix[new_st] + prefix[st] - A[new_st]
                    ans = min(ans, end - new_st + 1)
                    st = new_st + 1

            end += 1
        if ans == len(A) + 1:
            return -1
        return ans

    def splitArray(self, nums, m: int) -> int:
        total = sum(nums)
        left = math.ceil(total / m)
        print(left)
        right = total + 1
        ans = 9999999999
        while left != right:
            mid = left + (right - left) / 2
            n = 0
            index = 0
            too_large = False
            too_small = False
            tmp_max = 0
            while n != m - 1:
                chunk = 0
                while index < len(nums) and chunk + nums[index] < mid:
                    chunk += nums[index]
                    index += 1
                n += 1
                tmp_max = max(tmp_max, chunk)

                if index == len(nums):
                    too_large = True
                    break
            last_chunk = sum(nums[index:])
            tmp_max = max(tmp_max, last_chunk)
            ans = min(ans, tmp_max)
            if too_large:
                right = mid
            elif last_chunk < mid:
                right = mid
            elif last_chunk > mid:
                left = mid + 1

            elif last_chunk == mid:
                return last_chunk
        return ans

    def findAllConcatenatedWordsInADict(self, words):
        minLen = 999999
        maxLen = 0
        wordSet = set()

        def helper(word, index, minLen, maxLen, wordSet):

            if index >= len(word):
                return True
            if len(word) - index < minLen:
                return False
            for i in range(index + minLen, min(index + maxLen + 1, len(word) + 1)):

                if word[index:i] in wordSet and not (index == 0 and i == len(word)):

                    if helper(word, i, minLen, maxLen, wordSet):
                        return True
            return False

        for word in words:
            minLen = min(minLen, len(word))
            maxLen = max(maxLen, len(word))
            wordSet.add(word)

        ans = []
        for word in words:
            if helper(word, 0, minLen, maxLen, wordSet):
                ans.append(word)
        return ans

    def maximumMinimumPath(self, A) :
        def checker(A, t):
            stack = [(0, 0)]
            visited = {(0, 0)}
            neighbours = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            valid = False
            score = 99999999999
            while len(stack) > 0:
                top = stack.pop()
                # print(top)
                i, j = top
                if i < 0 or j < 0 or i >= len(A) or j >= len(A[0]):
                    continue
                if A[i][j] < t:
                    continue
                score = min(score, A[i][j])
                if i == len(A) - 1 and j == len(A[0]) - 1:
                    valid = True
                    break
                for neighbour in neighbours:
                    index = (i + neighbour[0], j + neighbour[1])
                    if index not in visited:
                        visited.add(index)
                        stack.append(index)
            return valid, score

        li = []
        for row in A:
            for item in row:
                li.append(item)
        left = min(li)
        right = max(li) + 1
        ans = -1
        while left < right:

            mid = left + (right - left) // 2
            print(left, right)
            valid, score = checker(A, mid)
            if valid:
                ans = max(ans, score)
                left = mid + 1
            else:
                right = mid
        return ans

    def equationsPossible(self, equations) -> bool:
        u = Union()
        for equation in equations:
            if equation[1] == "=":
                u.union(equation[0], equation[-1])

        for equation in equations:
            if equation[1] == "!":

                if u.find(ord(equation[0])) == u.find(ord(equation[-1])):
                    return False
        return True

    def alienOrder(self, words) -> str:
        dic = defaultdict(set)
        # for word in words:
        #     for i in range(len(word) - 1,-1,-1):
        #         for j in range(0,i):
        #             if word[i] != word[j]:
        #                 dic[word[i]].add(word[j])
        for i in range(len(words) - 1, 0, -1):
            word1 = words[i]
            for j in range(0, i):
                word2 = words[j]
                if word2 == word1:
                    continue
                for i in range(min(len(word1), len(word2))):
                    if word1[i] != word2[i]:
                        dic[word1[i]].add(word2[i])
                        break

        ans = []

        def dfs(visited, c, graph, seq):
            if c in visited:
                return False
            visited.add(c)

            for dependency in graph[c]:
                if not dfs(visited, dependency, graph, seq):
                    return False
            visited.remove(c)
            seq.append(c)
            return True

        print(dic)

        for word in words:

            for c in word:
                # print(ans)
                print(c)
                if c in ans:
                    continue
                visited = set()
                seq = []
                if dfs(visited, c, dic, seq):

                    for item in seq:
                        if item not in ans:
                            ans.append(item)
                else:
                    return ""

        return "".join(ans)

    def braceExpansionII(self, expression):
        stack, res, cur = [], [], []
        for i in range(len(expression)):
            v = expression[i]
            if v.isalpha():
                cur = [c + v for c in cur or ['']]
            elif v == '{':
                stack.append(res)
                stack.append(cur)
                res, cur = [], []
            elif v == '}':
                pre = stack.pop()
                preRes = stack.pop()
                cur = [p + c for c in res + cur for p in pre or ['']]
                res = preRes
            elif v == ',':
                res += cur
                cur = []
        return sorted(set(res + cur))
    def target(self,arr,target):
        ptr = len(arr) - 1
        for i in range(len(arr) - 1,-1,-1):
            if arr[i] != target:
                if i != ptr:
                    arr[ptr] = arr[i]
                ptr -= 1
        for i in range(ptr + 1):
            arr[i] = target
    def partition(self,arr,i,j):

    def quick_sort(self,arr,i,j):
        pivot = self.partition(arr,i,j)
        self.quick_sort(arr,i,pivot - 1)
        self.quick_sort(arr,pivot,j)

class Union:
    def __init__(self):
        self.rk = [1] * 26
        self.pr = [i for i in range(97,97 + 26)]
    def find(self,x):
        print(x)
        if self.pr[x -  97] != x:
            print(self.pr[x -  97],x)
            self.pr[x - 97] = self.find(self.pr[x - 97])
        return self.pr[x - 97]
    def union(self,x,y):
        xroot = self.find(ord(x))
        yroot = self.find(ord(y))
        if self.rk[xroot - 97] > self.rk[yroot - 97]:
            xroot,yroot = yroot,xroot
        if self.rk[xroot - 97] == self.rk[yroot - 97]:
            self.rk[yroot - 97] += 1
        self.pr[xroot - 97] = yroot
import heapq
import math
from collections import defaultdict
import threading
class Test:
    def __init__(self):
        self.count = 0
    def foo(self):
        if count == 0:
            print("a")


import threading


class H2O:
    def __init__(self):
        self.hcount = 0
        self.ocount = 0
        self.mutex = threading.Lock()
        self.ocond_release = threading.Condition(self.mutex)
        self.ocond_addh = threading.Condition(self.mutex)
        self.hcond_release = threading.Condition(self.mutex)
        self.hcond_addo = threading.Condition(self.mutex)

    def hydrogen(self, releaseHydrogen):

        self.mutex.acquire()
        while self.ocount >= 2:
            self.hcond_addo.wait()
        self.ocount += 1
        if self.ocount == 2:
            self.ocond_release.notify()
        while self.hcount == 0:
            self.hcond_release.wait()
        releaseHydrogen()
        self.hcount -= 1
        if self.hcount == 0:
            self.ocond_addh.notify()
        self.mutex.release()

        # releaseHydrogen() outputs "H". Do not change or remove this line.

    def oxygen(self, releaseOxygen):
        self.mutex.acquire()
        while self.hcount != 0:
            self.ocond_addh.wait()
        self.hcount = 2
        self.hcond_release.notify()
        self.hcond_release.notify()
        while self.ocount != 2:
            self.ocond_release.wait()
        releaseOxygen()
        #print("o good")
        self.ocount = 0
        self.hcond_addo.notify()
        self.hcond_addo.notify()
        self.mutex.release()


        # releaseOxygen() outputs "O". Do not change or remove this line.
def releaseHydrogen():
    print("H")
def releaseOxygen():
    print("O")
def setbit_helper(A,root):
    if root >= len(A) or A[root] == 1:
        return
    A[root] = 1
    setbit_helper(A,root * 2 + 1)
    setbit_helper(A, root * 2 + 2)
def set_bit(A,offset,length):
    if offset + len - 1 >= len(A) or offset < 0 or length < 0:
        return
    for i in range(offset,min(len(A),offset + length)):
        if A[i] == 1:
            continue
        else:
            setbit_helper(A,i)
        root = i

        while root != 0 and ((root % 2 == 0 and A[root - 1] == 1) or (root % 2 == 1 and root + 1 == 1)):
            root = (root - 1) // 2
            A[root] = 1
    print(A)
from collections import deque
class MySet:
    def __init__(self,load=0.75):
        self.pool = [[] for i in range(16)]
        self.size = 0
        self.load = load
        self.capacity = 16
    def __rehash(self):
        self.capacity *= 2
        new_pool = [[] for i in range(self.capacity)]
        for li in self.pool:
            for element in li:
                code = hash(element)
                new_pool[code % len(new_pool)].append(element)
        self.pool = new_pool
    def add(self,element):
        if self.size * self.load >= self.capacity:
            self.__rehash()
        code = hash(element) % self.capacity
        if self.contains(element):
            return
        self.pool[code].append(element)
    def contains(self,element):
        code = hash(element) % self.capacity
        if not self.pool[code]:
            return False
        else:
            for e in self.pool[code]:
                if element == e:
                    return True
            return False
    def delete(self,element):
        if not contains(element):
            return False
        else:
            self.pool[hash(element)].remove(element)
    def clear(self):
        self.pool = [[] for i in range(16)]
    def iterate(self):
        for d in self.pool:
            for element in d:
                print(element)

def check_square(li):
    for p1 in li:
        for p2 in li:
            line = [p1,p2]
            line.sort()
            m = p2[1] - p1[1]
            prependiculat_m = (-1/m)
            prependiculat_m * p1
def checker(num):
     num = str(num)
     count = Counter(num)
     for i in range(len(num)):
         if count[str(i)] != int(num[i]):
             return False
     return True

def self_descriptive():
    def dfs(node,n,ans):
        total = sum(node)
        if len(node) == n:
            candidate = "".join([str(ele) for ele in node])
            if checker(candidate):
                ans.append(int(candidate))
            node.pop()
            return
        for c in range(0,n - total + 1):
            if len(node) == 0 and c == 0:
                continue
            node.append(c)
            dfs(node,n,ans)
        if len(node) > 0:
            node.pop()
        return
    ans = []
    for i in range(1,10):
        dfs([],i,ans)

    print(ans)
if __name__ == "__main__":
    s = Solution()
    s.braceExpansionII("{{a,b}{a,b}")
    stack = [1,2]

    #stack = [1, 2, 4, 2, 5, 7, 3, 7, 3, 5]
    s.target(stack,5)
    print(stack)
    self_descriptive()


    # threading.Thread(target=h20.hydrogen, args=(releaseHydrogen,)).start()
    # threading.Thread(target=h20.hydrogen, args=(releaseHydrogen,)).start()
    # threading.Thread(target=h20.hydrogen, args=(releaseHydrogen,)).start()
    # threading.Thread(target=h20.hydrogen, args=(releaseHydrogen,)).start()
    # threading.Thread(target=h20.hydrogen, args=(releaseHydrogen,)).start()
    # threading.Thread(target=h20.hydrogen, args=(releaseHydrogen,)).start()
    # threading.Thread(target=h20.hydrogen, args=(releaseHydrogen,)).start()
    #
    # threading.Thread(target=h20.hydrogen, args=(releaseHydrogen,)).start()
    # threading.Thread(target=h20.oxygen, args=(releaseOxygen,)).start()
    # threading.Thread(target=h20.oxygen, args=(releaseOxygen,)).start()
    # threading.Thread(target=h20.hydrogen, args=(releaseHydrogen,)).start()
    # threading.Thread(target=h20.oxygen, args=(releaseOxygen,)).start()
