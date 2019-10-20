
import java.io.IOException;
import java.lang.reflect.Array;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.math.BigInteger;

import java.util.*;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

import static java.util.Arrays.asList;

/**
 * Created by jiazheng on 1/25/18.
 */
class Solution extends Thread {
    public List<String> summaryRanges(int[] nums) {

        String arrow = "->";
        List<String> list = new ArrayList<String>();
        for (int i = 0; i < nums.length; i++) {
            int start = nums[i];
            int j = 1;
            for (; j + i < nums.length; j++) {
                if (nums[i + j] - 1 == start) {
                    start = nums[i + j];
                }
            }
            if (start == nums[i]) {
                list.add(Integer.toString(start));
            } else {
                String temp = Integer.toString(nums[i]) + arrow + Integer.toString(start);
                list.add(temp);
            }
            i += j - 1;

        }
        return list;
    }
    public int maximumGap(int[] nums) {

        if(nums.length < 2) return 0;
        Integer min= nums[0];
        Integer max = nums[0];
        for(Integer i:nums){
            if(i < min){
                min = i;
            }
            if(i > max){
                max = i;
            }
        }

        return 0;
    }
    public int numSquares(int n) {
        int[] arr = new int[n + 1];
        arr[0] = 0;
        arr[1] = 1;
        if(n < 2) return n;
        for(int i = 2; i <= n;i++){
            int min = 9999999;
            for(int j = 1; j*j <= i;j++){

                if(arr[i -j*j] < min )
                    min = arr[i - j*j];
            }
            arr[i] = min + 1;
        }
        return arr[n];
    }
    public void setZeroes(int[][] matrix) {
        boolean lastZero = false;
        if(matrix.length == 0 || matrix[0].length == 0) return;
        int rowNum = matrix.length;
        int colNum = matrix[0].length;
        for(int i = 0; i < rowNum;i++){
            if(matrix[i][colNum - 1] == 0)
                lastZero = true;
        }
        //Arrays.sort();
        for(int i = 0; i < rowNum;i++){
            for(int j = 0; j < colNum - 1;j++){
                if(matrix[i][j] == 0){
                    matrix[i][colNum - 1] = 0;
                    for(int k = 0; k < colNum - 1;k++){
                        if(matrix[i][k]!=0){
                            matrix[i][k] = 0;

                        }else{
                            matrix[i][k] = 1;
                        }
                    }
                }
            }
        }
        for(int i = 0; i < rowNum;i++){
            if(matrix[i][colNum - 1] == 0){
                for(int j = 0; j < colNum - 1;j++){
                    if(matrix[i][j] == 1){
                        for(int k = 0; k < rowNum;k++){
                            matrix[k][j] = 0;
                        }
                    }
                }
            }
        }
        if(lastZero){
            for(int i = 0; i < rowNum;i++){
                matrix[i][colNum - 1] = 0;
            }
        }
    }

    public String removeDuplicateLetters(String s) {
        //Set<Character> set = new HashSet<Character>();

        List<Character> list = new LinkedList<>();

        for(int i = 0; i < s.length();i++){
            int index = list.indexOf(s.charAt(i));
            if(index == -1){
                list.add(s.charAt(i));
            }else{
                if(list.size() == 1 || index == list.size() - 1)
                    continue;
                if(list.get(index + 1) < list.get(index)){
                    list.add(list.remove(index));
                }
            }
        }
        StringBuffer sb = new StringBuffer("");
        for(int i = 0;i < list.size();i++){
            sb.append(list.get(i));
        }
        return sb.toString();

    }
    public int minimumTotal(List<List<Integer>> triangle) {
        if(triangle == null || triangle.isEmpty() || triangle.get(0) == null || triangle.get(0).isEmpty()) return 0;
        if(triangle.size() == 1)
            return triangle.get(0).get(0);
        List<Integer> list = new ArrayList<>();
        list.add(triangle.get(0).get(0));
        for(int i = 1; i < triangle.size();i++){
            List<Integer> templist = new ArrayList<>();
            for(Integer j: triangle.get(i)){
                int min = 999999999;
                for(Integer k:list){
                    if(k + j < min)
                        min = k + j;
                }
                templist.add(min);
            }
            list = templist;
        }
        int ans = list.get(0);
        for(int i = 0; i < list.size();i++){
            if(list.get(i) < ans )
                ans = list.get(i);
        }
        return ans;

    }
    char direction = '0';
    public boolean circularArrayLoop(int[] nums) {
        for(int i = 0; i < nums.length;i++){
            if(loop(nums,i))
                return true;
        }return false;
    }
    public boolean loop(int[] nums,int start) {
        if(nums.length == 0) return false;
        int slow = 0;
        int fast = 0;
        slow = move(start,nums,1,true);
        fast = move(start,nums,2,false);
        if(slow ==  start) return false;
        if(slow == -1 || fast == -1) return false;
        while((fast != nums.length - 1) && (slow != nums.length - 1)){

            if(slow == fast)
                return true;
            int tempslow = move(slow,nums,1,false);
            int tempfast = move(fast,nums,2,false);
            if(tempslow == -1 || tempfast == -1 || slow == tempslow ) return false;
            slow = tempslow;
            fast = tempfast;
        }
        return false;

    }
    private int move(int index,int[] nums,int steps,boolean first){
        for(int i = 0; i < steps;i++){
            if(first){
                direction = nums[index] < 0?'b':'f';
            }
            if(direction == 'f' && nums[index] <= 0) return -1;
            if(direction == 'b' && nums[index] >= 0) return -1;
            if(0 > index + nums[index]){
                index = nums.length - Math.abs(index + nums[index]%nums.length);
                if(index == nums.length)
                    index = 0;
            }else if(index + nums[index] >= nums.length){
                index =  (index + nums[index])%nums.length;
            }else{
                index = index + nums[index];
            }
        }
        return index;
    }

        public void rotate(int[] nums, int k) {
            k = k % nums.length;
            int count = 0;
            for(int i = 0; count < nums.length;i++){
                int next_index = (i + k) % nums.length;
                //System.out.println(i);
                int cur = nums[i];
                do{
                    //int cur = nums[next_index];
                    System.out.println(next_index);
                    int next = nums[next_index];
                    nums[next_index] = cur;
                    cur = next;
                    next_index = (next_index + k) % nums.length;
                    count++;

                    if(count >= nums.length){

                        System.out.println(i);
                        break;
                    }
                }while(i != next_index);
                nums[next_index] = cur;
                count++;
            }
            return;
        }
    public void gameOfLife(int[][] board) {
        for(int i = 0; i < board.length;i++){
            for(int j = 0; j < board[0].length;i++){
                explore(board,i,j);
            }
        }
        for(int i = 0 ; i < board.length; i++){
            for(int j = 0; j < board[0].length;j++){
                if(board[i][j] == -1)
                    board[i][j] = 0;
                else if(board[i][j] == -2){
                    board[i][j] = 1;
                }
            }
        }

    }
    private void explore(int[][] board, int rownum,int colnum){
        int num_1 = 0;
        for(int i = -1; i < 2; i++){
            if( board.length <=rownum + i || rownum + i <  0)
                continue;
            for(int j = -1; j < 2; j++){
                if( board[0].length <=colnum + j || colnum + j <  0)
                    continue;
                if(board[i][j] == 1 || board[i][j] == -1)
                    num_1++;
            }
        }
        if(board[rownum][colnum] == 1){
            if(num_1 < 2)
                board[rownum][colnum] = -1;
            else if(num_1 > 3){
                board[rownum][colnum] = -1;
            }
        }else{
            if(num_1 == 3)
                board[rownum][colnum] = -2;
        }
    }
    public int[] singleNumber(int[] nums) {
        int aggregation = 0;
        for(int i = 0; i < nums.length;i++){
            aggregation ^= nums[i];
        }
        int judge = 1;
        for(int i = 0; i < 32;i ++ ){
            if((aggregation & judge) == judge)
                break;
            judge <<= 1;

        }

        ArrayList<Integer> list1 = new ArrayList<>();
        ArrayList<Integer> list2 = new ArrayList<>();
        for(int i = 0; i < nums.length;i++) {
            if ((nums[i] & judge) == judge) {
                list1.add(nums[i]);
            } else {
                list2.add(nums[i]);
            }
        }
        int[] result = new int[2];
        int first = 0;
        for(Integer i:list1){
            first ^= i;
        }
        int second = 0;
        for(Integer i:list2){
            second ^= i;
        }
        result[0] = first;
        result[1] = second;
        return result;
    }
    public int maxCoins(int[] nums) {
        List<Integer> list = Arrays.stream(nums).boxed().collect(Collectors.toList());
        list.add(0,1);
        list.add(1);
        int n = nums.length;
        int[][] arr = new int[n + 2][n + 2];
        for(int l = 1; l <= n;l++){
            for(int i = 1; i <= n + 1 - l; i++ ){
                int j = i + l - 1;
                for(int k = i ; k <= j;k++){
                   int temp = Math.max(arr[i][j],arr[i][k - 1] + list.get(k)*list.get(i - 1)*list.get(j + 1)+arr[k + 1][j]);
                   arr[i][j] = temp;
                }
            }
        }
        return arr[1][n];
    }
    int target = 0;
    ArrayList<List<Integer>> result_list = new ArrayList<>();
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        this.target = target;
        helper(candidates,0,new ArrayList<Integer>(),0);
        return result_list;
    }
    private void helper(int[] candidates,int start,ArrayList<Integer> list,int sum){
        for(int i = start; i < candidates.length;i++){
            int result = sum + candidates[i];
            if(result == target){
                ArrayList<Integer> templist = new ArrayList<>();
                templist.addAll(list);
                templist.add(candidates[i]);
                result_list.add(templist);
                continue;
            }
            if(result > target){
                continue;
            }
            list.add(candidates[i]);
            helper(candidates,i,list,sum + candidates[i]);
        }
        if(!list.isEmpty())
            list.remove(list.size() - 1);
    }
    public int findComplement(int num) {
        int ruler = 1<<30;
        int index = 30;
        for(; index >= 0;index--){
            if((ruler & num) == ruler)
                break;
            ruler >>= 1;
        }

        int zero = 0;
        int one = 1;
        int max = Integer.MAX_VALUE;

        for(int i = 0; i <= index ;i++){
            if((num&one) == one){
                num &= (max - one);
            }else{
                num |= one;
            }
            one <<= 1;
        }
        return num;
    }
    public int minPathSum(int[][] grid) {
        if(grid == null || grid.length == 0 || grid[0].length == 0)
            return 0;
        int[][] map = new int[grid.length][grid[0].length];
        map[0][0] = grid[0][0];
        for(int i = 1; i < grid.length;i++){
            map[i][0] = grid[i][0] + map[i - 1][0];
        }
        for(int j = 1; j < grid[0].length;j++){
            map[0][j] = grid[0][j] + map[0][j - 1];
        }
        for(int i = 1; i < grid.length; i ++){
            for(int j = 1; j < grid[0].length;i++){
                map[i][j] = Math.min(map[i-1][j],map[i][j-1]) + grid[i][j];
            }
        }
        return map[ grid.length - 1][grid[0].length - 1];

    }
    public int maxProfit(int k, int[] prices) {
        int len = prices.length;
        if (k >= len / 2) return quickSolve(prices);

        int[][] t = new int[k + 1][len];
        for (int i = 1; i <= k; i++) {
            int tmpMax =  -prices[0];
            for (int j = 1; j < len; j++) {
                t[i][j] = Math.max(t[i][j - 1], prices[j] + tmpMax);
                tmpMax =  Math.max(tmpMax, t[i - 1][j - 1] - prices[j]);
            }
        }
        return t[k][len - 1];
    }


    private int quickSolve(int[] prices) {
        int len = prices.length, profit = 0;
        for (int i = 1; i < len; i++)
            if (prices[i] > prices[i - 1]) profit += prices[i] - prices[i - 1];
        return profit;
    }
    public int longestConsecutive(int[] nums) {
        HashMap<Integer,Integer> map = new HashMap<Integer,Integer>();
        int max = 0;
        for(int i = 0; i < nums.length;i++){
            if(!map.containsKey(nums[i])){
                if(!map.containsKey(nums[i] - 1) && !map.containsKey(nums[i] + 1)) {
                    map.put(nums[i], 1);
                    if(max == 0) max = 1;
                }
                else if(map.containsKey(nums[i] - 1) && !map.containsKey(nums[i] + 1)){
                    int dist = map.get(nums[i] - 1);
                    map.computeIfPresent(nums[i] - dist,(key,value)->dist + 1);
                    map.put(nums[i],dist + 1);
                    max = Math.max(max,dist + 1);
                }
                else if(!map.containsKey(nums[i] - 1) && map.containsKey(nums[i] + 1)){
                    int dist = map.get(nums[i] + 1);
                    map.computeIfPresent(nums[i] + dist,(key,value)->dist + 1);
                    map.put(nums[i],dist + 1);
                    max = Math.max(max,dist + 1);
                }else{
                    int dist1 = map.get(nums[i] - 1);
                    int dist2 = map.get(nums[i] + 1);
                    map.computeIfPresent(nums[i] + dist2,(key,value)->dist2 + 1 + dist1);
                    map.computeIfPresent(nums[i] - dist1,(key,value)->dist2 + 1 + dist1);
                    max = Math.max(max,dist1 + 1 + dist2);
                    map.put(nums[i],0);
                }
            }
        }
        return max;
    }
    public int findShortestSubArray(int[] nums) {
        HashMap<Integer,Integer> map = new HashMap<Integer,Integer>();
        HashMap<Integer,ArrayList<Integer>> map1 = new HashMap<Integer,ArrayList<Integer>>();

        for(int i = 0; i < nums.length;i++){
            map.computeIfPresent(nums[i],(key,value)->value + 1);
            map.computeIfAbsent(nums[i],(key)->1);
        }
        int max = 0;
        for(Map.Entry<Integer,Integer> e: map.entrySet()){
            if(e.getValue() > max)
                max = e.getValue();
        }
        for(Map.Entry<Integer,Integer> e: map.entrySet()){
            if(e.getValue() == max)
                map1.put(e.getKey(),new ArrayList<Integer>());
        }
        for(int i = 0; i < nums.length;i++){
            if(map1.containsKey(nums[i]) && map1.get(nums[i]).size() == 0){
                map1.get(nums[i]).add(i);
            }else if(map1.containsKey(nums[i]) && map1.get(nums[i]).size() == 1){
                map1.get(nums[i]).add(i);
            }else if(map1.containsKey(nums[i]) && map1.get(nums[i]).size() == 2){
                map1.get(nums[i]).remove(1);
                map1.get(nums[i]).add(i);
            }
        }
        int ans = 999999;
        for(Map.Entry<Integer,ArrayList<Integer>> e: map1.entrySet()){
            if(e.getValue().get(e.getValue().size() - 1) - e.getValue().get(0) < ans){
                ans = e.getValue().get(e.getValue().size() - 1) - e.getValue().get(0) + 1;
            }
        }
        return ans;

    }
    public List<List<Integer>> palindromePairs(String[] words) {
        ArrayList<List<Integer>> ans = new ArrayList<>();
        HashMap<String,Integer> map = new HashMap<String,Integer>();
        for(int i = 0;i < words.length;i++){
            map.put(new StringBuilder(words[i]).reverse().toString(),i);
        }
        for(int i = 0; i < words.length;i++){
            for(int j = 0;j < words[i].length();j++){
                if(map.containsKey("")){
                    ArrayList<Integer> temp = new ArrayList<>();
                    temp.add(map.get(""));
                    temp.add(i);
                    ans.add(temp);
                    ArrayList<Integer> temp1 = new ArrayList<>();
                    temp1.add(i);
                    temp1.add(map.get(""));
                    ans.add(temp1);
                    continue;
                }
                String suffix = words[i].substring(j);
                String prefix = words[i].substring(0,j);

                if(map.containsKey(suffix) && isPalindrome(prefix) && map.get(suffix) != i){
                    ArrayList<Integer> temp = new ArrayList<>();
                    temp.add(map.get(suffix));
                    temp.add(i);
                    ans.add(temp);
                }
                if(map.containsKey(prefix) && isPalindrome(suffix) && map.get(prefix) != i){
                    ArrayList<Integer> temp = new ArrayList<>();
                    temp.add(i);
                    temp.add(map.get(prefix));
                    ans.add(temp);
                }
            }
        }
        return ans;
    }
    private boolean isPalindrome(String str){
        int start = 0;
        int end = str.length() - 1;
        while(end > start){
            if(str.charAt(start) != str.charAt(end))
                return false;
            start++;
            end--;
        }

        return true;
    }
    public List<Integer> findDuplicates(int[] nums) {
        int marker = 0;
        ArrayList<Integer> list = new ArrayList<Integer>();
        for(int i = 0; i < nums.length;i++){
            if(nums[i] == i + 1){
                nums[i] = -1;
            }else if(nums[i] == -1 || nums[i] == -2)
                continue;
            else{
                int temp = nums[i];
                nums[i] = 0;
                while(temp > 0){

                    int next = nums[temp - 1];

                    if(next <= 0){
                        nums[temp - 1]--;
                        temp = nums[temp - 1];
                    }else{
                        nums[temp - 1] = -1;

                        temp = next;
                    }
                }
            }
        }
        for(int i = 0; i < nums.length;i++){
            if(nums[i] == -2 )
                list.add(i + 1);
        }
        return list;
    }
    boolean [][] gold = new boolean[9][9];
    public void solveSudoku(char[][] board) {
        for(int i = 0; i < 9;i++){
            for(int j = 0; j < 9;j++){
                gold[i][j] = false;
                if(board[i][j]!='.')
                    gold[i][j] = true;
            }
        }
        if(board[0][0] == '.'){
            for(int i = 1; i <10;i++ ){
                helper(board,0,Character.toChars(48 + i)[0]);
            }
        }
        else
            helper(board,0,board[0][0]);

        return;
    }
    boolean finished = false;
    private void helper(char[][] board,int start,char num){
        //if(start == 25 && board[0][2] == '4' && board[0][3] == '6' && board[0][6] == '9' && board[1][1] == '7' & board[2][3] == '3'  )
          //  System.out.print("aasa");
        if (finished){
            return;
        }
        board[start/9][start%9] = num;
        if(gold[start/9][start%9] ){

        }
        else if(!check(board,start/9,start%9)){
            board[start/9][start%9] = '.';
            return;
        }

        if(start == 80){
            finished = true;
            return;
        }

        for(int i = start; i < 81;i++){
            if(board[(i + 1)/9][(i + 1)%9] == '.'){
                for(int j = 1; j < 10;j++){
                    helper(board,i + 1,Character.toChars(48 + j)[0]);
                    if(finished)
                        return;
                }

            }
            else{
                helper(board,i + 1,board[(i + 1)/9][(i + 1)%9]);
                if(finished)
                    return;
            }
            if( !gold[i/9][i%9]) {
                board[i/9][i%9] = '.';
            }
            return;
        }


    }
    private boolean check(char[][] board,int row,int col){
        if(board[row][col] == '.') return true;
        int[] arr_row = new int[9];
        int[] arr_col = new int[9];
        int[] arr_box = new int[9];
        for(int i = 0; i < 9;i++){
            if(! (board[row][i] == '.')){

                int num = board[row][i] - '0';
                try {
                    arr_row[num - 1] += 1;
                }
                catch (Exception e){
                    System.out.println(e);
                }
            }
            if(!(board[i][col] == '.')){

                char temp = board[i][col];
                try {
                    arr_col[board[i][col] - '0' - 1] += 1;
                }catch (Exception e){
                    System.out.println("aaa");
                }
            }
        }

        int r = (row / 3) *3;
        int c = (col / 3) * 3;
        for(int i = 0; i < 3;i++){
            for(int j = 0; j < 3;j++){
                if(board[r + i][c + j] == '.')
                    continue;

                arr_box[board[r + i][c + j] - '0' - 1] += 1;
            }
        }
        for(int i = 0; i < 9;i++){
            if(arr_col[i] > 1 || arr_row[i] > 1 || arr_box[i] > 1)
                return false;
        }
        return true;

    }
    String longestPalindrome(String s) {
        StringBuffer[][] sb = new StringBuffer[s.length()][s.length()];
        for(int i = 0; i < s.length();i++){
            for(int j = 0; j < s.length();j++){
                if(j == i + 1){

                }
                else
                    sb[i][j] = new StringBuffer(".");
            }
            sb[i][i] = new StringBuffer("");
        }
        for(int i = 0; i < s.length();i++){
            for(int j = 0;j <= i;j++){

            }
        }
        return "";
    }
    public int canCompleteCircuit(int[] gas, int[] cost) {
        HashMap<Integer,ArrayList<Integer>> map = new HashMap<Integer,ArrayList<Integer>>();
        int[] gain = new int[gas.length];
        for(int i = 0; i < gas.length;i++){
            gain[i] = gas[i] - cost[i];
        }
        for(int i = 0; i < gas.length;i++){
            if(gain[i] < 0) continue;
            int j = i;
            int total = 0;
            int end = -1;
            for(;  !(j % gas.length == i && j >= gas.length);j++){
                if(!map.containsKey(j)){
                    total+= gain[j%gas.length];
                    if(total < 0){
                        ArrayList<Integer> temp = new ArrayList<>();
                        temp.add(j);
                        temp.add(total);
                        map.put(i,temp);
                        end = j;
                        break;

                    }
                }else{
                    total += map.get(j).get(1);
                    if(total < 0){
                        ArrayList<Integer> temp = new ArrayList<>();
                        temp.add(map.get(j).get(0));
                        temp.add(total);
                        map.put(i,temp);
                        end = map.get(j).get(0);
                        break;
                    }else{
                        j = map.get(j).get(0);
                    }
                }
            }
            if(j >= gas.length && j % gas.length == i)
                return i;
            int new_i = (end + 1)% gas.length;
            if(new_i < i)
                return -1;
        }
        return -1;
    }

    public int calculate(String s) {
        int ans = 0;
        Stack<Character> stack = new Stack<>();
        StringBuffer sb = new StringBuffer("");
        for(int i = 0; i < s.length();i++){
            if(s.charAt(i)!=' '){
                sb.append(s.charAt(i));
            }
        }
        s = sb.toString();
        for(int i = 0; i < s.length();i++){
            if(s.charAt(i)!=')'){
                stack.add(s.charAt(i));
            }else{
                Stack<Character> temp = new Stack<Character>();
                temp.add('+');
                while(stack.peek()!='('){
                    temp.add(stack.pop());
                }
                stack.pop();
                StringBuffer num = new StringBuffer("");

                Integer sum = 0;
                char sign = '+';
                while(!temp.empty()){
                    char c = temp.pop();
                    if((temp.empty()||(temp.peek()=='+' || temp.peek()=='-') )){
                        int value = 0;

                        value = Integer.valueOf(num.toString());
                        if(sign == '+')
                            sum += value;
                        else
                            sum -= value;
                        sign = c;
                        num = new StringBuffer("");
                        num.append(c);
                    }
                    else if((c!='-' ) && c!='+')
                        num.append(c);
                    else{
                        int value = 0;

                            value = Integer.valueOf(num.toString());
                        if(sign == '+')
                            sum += value;
                        else
                            sum -= value;
                        sign = c;
                        num = new StringBuffer("");
                    }
                }


                String finalsum = sum.toString();
                int k = 0;

                for(; k < finalsum.length();k++){
                    stack.add(finalsum.charAt(k));
                }
            }
        }
        Stack<Character> temp = new Stack<Character>();
        temp.add('+');
        while(!stack.empty()){
            temp.add(stack.pop());
        }

        StringBuffer num = new StringBuffer("");
        Integer sum = 0;
        char sign = '+';
        while(!temp.empty()){
            char c = temp.pop();
            if(c!='-' && c!='+')
                num.append(c);
            else{

                    int value;

                        value = Integer.valueOf(num.toString());
                if(sign == '+')
                    sum += value;
                else
                    sum -= value;
                sign = c;
                num = new StringBuffer("");
            }
        }
        return sum;



    }
    void modify(Integer i){
        AtomicInteger it = new AtomicInteger(9);

    }
    class Trie{
        int self= -1;
        int count = 0;
        Trie[] child = null;
        public Trie(){
            child = new Trie[2];
        }
    }
    public int findMaximumXOR(int[] nums) {
        Trie root = new Trie();

        int judge = 1<<30;
        for(int i = 0; i < nums.length;i++){
            Trie temp = root;
            for(int j = 0; j < 31;j ++){
                if((judge & nums[i]) == judge){
                    if(temp.child[1] == null){
                        temp.child[1] = new Trie();
                    }
                    temp = temp.child[1];
                    if(temp.self == -1){
                        temp.self = 1;
                    }
                    temp.count += 1;
                }else{
                    if(temp.child[0] == null){
                        temp.child[0] = new Trie();
                    }


                    temp = temp.child[0];
                    if(temp.self == -1){
                        temp.self = 0;
                    }
                    temp.count += 1;
                }
                judge >>= 1;
            }
            judge = 1<<30;
        }
        Trie temp = root;


        int max = 0;
        for(int i = 0; i < nums.length;i++){
            int sum = 0;
            temp = root;
            for(int j = 30; j >=0 ;j--){
                if((judge & nums[i]) == judge){
                    if(temp.child[0] != null){
                        sum+=Math.pow(2,j);
                        temp = temp.child[0];
                    }else{
                        temp = temp.child[1];
                    }
                }else{
                    if(temp.child[1] != null){
                        sum+=Math.pow(2,j);
                        temp = temp.child[1];
                    }else{
                        temp = temp.child[0];
                    }
                }
                judge >>= 1;
                if(temp == null)
                    break;
            }
            judge = 1<<30;
            max = Math.max(max,sum);
        }
        return max;
    }
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        if(nums1.length == 0 && nums2.length == 0)
            return 0;
        if(nums1.length == 0 && nums2.length == 0)
            return 1;
        if(nums1.length > nums2.length || (nums1.length == nums2.length && nums1[0] > nums2[0])){
            int[] nums3 = nums2;
            nums2 = nums1;
            nums1 = nums3;
        }

        int mode = (nums1.length + nums2.length)%2;
        int total = 0;
        if(mode == 0){
            if(nums1.length == 0)
                return (double)(nums2[(nums2.length - 1)/2] + nums2[(nums2.length )/2])/2;
           total = ((nums1.length + nums2.length) - 1)/2;
        }else{
            if(nums1.length == 0)
                return nums2[(nums2.length )/2];
            total = (nums1.length + nums2.length)/2;
        }
        int begin = 0;
        int end = nums1.length;
        int index = -1;
        char type = 'a';
        while(begin < end){
            index = begin + (end - begin) / 2;
            int rest = total - index;
            if(nums2[rest] >= nums1[index]){
                if((rest - 1) < 0 || nums2[rest - 1] <= nums1[index]){
                    break;
                }
                else {
                    type = 's';
                    begin = index + 1;
                }
            }
            else if(nums2[rest] < nums1[index]){
                end = index;
                type = 'l';
            }
        }
        if(begin != end){
            if(mode == 1)
                return nums1[index];
            else{
                if(index != nums1.length - 1)
                    return ((double)(nums1[index]) + (double)(Math.min(nums2[total - index],nums1[index + 1])))/2;
                else
                    return ((double)(nums1[index]) + (double)(nums2[total - index]))/2;
            }
        }
        else{
            if(mode == 1){
                if(type == 's'){
                    return nums2[total - index - 1];
                }else{
                    return nums2[total - index];
                }
            }else{
                if(type == 's'){
                    if(index != nums1.length -1)
                        return ((double)(nums2[total - index - 1]) +(double) (Math.min(nums2[total - index ],nums1[index + 1])))/2;
                    else
                        return ((double)(nums2[total - index - 1]) + (double)(nums2[total - index ]))/2;
                }else{
                    return ((double)(nums2[total - index]) +(double) (Math.min(nums2[total - index + 1 ],nums1[index])))/2;
                }
            }
        }
    }
    public ListNode mergeKLists(ListNode[] lists) {
        if(lists.length == 0) return null;
        PriorityQueue<ListNode> queue = new PriorityQueue<>(new Comparator<ListNode>(){
            public int compare(ListNode o1,ListNode o2){
                if(o1.val > o2.val)
                    return 1;
                if(o1.val < o2.val)
                    return -1;
                return 0;
            }
        });
        for(int i = 0; i < lists.length;i++){
            queue.add(lists[i]);
        }
        ListNode dummy = new ListNode(0);
        ListNode root = dummy;
        while(!queue.isEmpty()){
            ListNode temp = queue.remove();
            root.next = temp;
            ListNode next =  temp.next;
            temp.next = null;
            if(next!=null)
                queue.add(next);
            root = root.next;
        }
        return dummy.next;

    }
    public List<List<Integer>> threeSum(int[] nums) {
        Arrays.sort(nums);
        ArrayList<List<Integer>> list = new ArrayList<>();
        for(int i = 0; i < nums.length- 2;i++){
            int needed = 0 - nums[i];
            int begin = i + 1;
            int end = nums.length - 1;

            ArrayList<Integer> temp = new ArrayList<Integer>();
            while(end < begin){
                if(nums[end] + nums[begin] == needed){
                    temp.add(nums[i]);
                    temp.add(nums[begin]);
                    temp.add(nums[end]);
                    while(begin + 1 < end && nums[begin + 1] == temp.get(1)){
                        begin++;
                    }
                    while(begin  < end - 1 && nums[end - 1] == temp.get(2)){
                        end--;
                    }
                    begin++;
                    end--;
                }else if(nums[end] + nums[begin] > needed){
                    end--;
                }else if(nums[end] + nums[begin] < needed){
                    begin++;
                }
            }
            while(temp.size() > 0 && i + 1 < nums.length && nums[i + 1] == temp.get(0)){
                i++;
            }
            if(temp.size() > 0)
                list.add(temp);
        }
        return list;
    }
    public int trap(int[] height) {
        if(height.length < 3) return 0;
        int leftmax = height[0];
        int left = 0;
        int right = height.length - 1;
        int rightmax = height[right];
        int sum = 0;
        int last = 0;
        while(left <= right){
            int level = Math.min(leftmax,rightmax);
            int lower = Math.min(height[left],height[right]);
            int leftadd = Math.max(level - height[left],0);
            int rightadd =  Math.max(level - height[right],0);
            if(left!=right)
                sum += (leftadd + rightadd);
            else sum+=leftadd;
            if(height[left] >= leftmax)
                leftmax = height[left];
            if(height[right] >= rightmax)
                rightmax = height[right];
            if(left + 1 == right)
                break;
            if(height[left] <= level) {
                left++;
            }
            if(height[right] <= level)
                right--;
        }
        return sum;
    }
    public int[] productExceptSelf(int[] nums) {
        int[] forward = new int[nums.length+1];
        int[] backward = new int[nums.length + 1];
        forward[0] = 1;
        backward[nums.length ] = 1;
        for(int i = 0; i < nums.length;i++){
            forward[i + 1] = forward[i]* nums[i];
            backward[nums.length - i - 1] = nums[i] * backward[nums.length - i];
        }
        int[] ans = new int[nums.length ];
        for(int i = 0; i< nums.length;i++){
            ans[i] = forward[i]*backward[i + 1];
        }
        return ans;
    }
    public int lengthLongestPath(String input) {
        String[] list = input.split("\n");
        int[] levelLength = new int[list.length + 1];
        int[] sum = new int[list.length + 1];
        int max = 0;
        for(int i = 0; i < list.length;i++){

            int level = 0;
            int j = 0;
            for(; j < list[i].length();j++){
                if(list[i].charAt(j) == '\t')
                    level++;
                else
                    break;
            }
            int index = list[i].indexOf('.');
            if(index != -1){
                max = Math.max(sum[level] + list[i].length(),max);
            }else{
                sum[level + 1] = sum[level] + (list[i].length()) - level;
                levelLength[level + 1] = list[i].length() - level;
            }
        }
        return max;

    }
    public int singleNumberII(int[] nums) {
        int[] arr = new int[32];
        for(int i = 0; i < nums.length;i++){
            int judge = 1;
            for(int j = 0; j < 31;j++){

                if((nums[i] & judge) == judge){
                    arr[j] = (arr[j] + 1)%3;
                }
                judge<<=1;
            }
            if(nums[i] < 0)
                arr[31] = (arr[31] + 1)%3;
        }
        int ans = 0;
        int judge = 1;
        for(int i = 0; i < 31;i++){
            if(arr[i] == 1)
                ans |= judge;
            judge<<=1;
        }
        if(arr[31] == 1)
            ans|=Integer.MIN_VALUE;
        return ans;


    }

    public int minDistance(String word1, String word2) {
        int[][] dp = new int[word2.length() + 1][word1.length() + 1];
        for(int i = 0; i < word2.length() + 1;i++){
            dp[i][0] = i;
        }
        for(int i = 0; i < word1.length() + 1;i++){
            dp[0][i] = i;
        }
        for(int i = 1; i < word2.length() + 1 ;i++){
            for(int j = 1; j < word1.length() + 1;j++){
                if(word2.charAt(i - 1) == word1.charAt(j - 1)){
                    dp[i][j] = dp[i - 1][j - 1];
                }
                else{
                    dp[i][j] = Math.min(dp[i - 1][j],dp[i][j - 1]);
                    dp[i][j] = Math.min(dp[i][j],dp[i - 1][j - 1]) + 1;
                }
            }
        }
        return dp[word2.length()][word1.length()];
    }
    public int longestValidParentheses(String s) {
        int count = 0;
        int sum = 0;
        int max = 0;
        int[] dp = new int[s.length() + 1];
        dp[0] = 0;
        for(int i = 0; i < s.length();i++){
            if(s.charAt(i) == ')') {
                count--;
                if (count < 0) {
                    count = 0;
                    dp[i + 1] = 0;
                    sum = 0;
                }else{
                    sum+=2;
                    dp[i + 1] = dp[i + 1 - sum] + sum;
                    sum = Math.max(sum,dp[i + 1]);
                }
            }
            else {
                count++;
                dp[i + 1] = 0;
                sum = 0;
            }
            max = Math.max(max, dp[i + 1]);

        }
        return max;

    }
    public int maxProduct(int[] nums) {
        if(nums.length == 0) return 0;
        if(nums.length == 1)return nums[0];
        int min = nums[0];
        int max = nums[0];
        int ans = nums[0];
        for(int i = 1; i < nums.length;i++){
            if(nums[i] < 0){
                int temp = min;
                min = max;
                max = temp;
            }
            min = Math.min(nums[i],min*nums[i]);
            max = Math.max(nums[i],max*nums[i]);
            ans = Math.max(ans,max);
        }
        return ans;
    }
    public int findMin(int[] nums) {
        if(nums.length == 0) return 0;
        if(nums.length == 1) return nums[0];
        int ruler = nums[nums.length - 1];
        int begin = 0;
        int end = nums.length;
        while(begin < end){
            int mid = begin + (end - begin)/2;
            if(mid == 0 && nums[0] < ruler) return nums[0];
            if(mid == nums.length - 1 && nums[mid - 1] > ruler) return nums[nums.length - 1];
            if(nums[mid - 1] > nums[mid] && nums[mid + 1] > nums[mid])
                return nums[mid];
            if(nums[mid] < ruler)
                end = mid;
            else if(nums[mid] == ruler){
                end = mid;
            }
            else
                begin = mid + 1;
        }
        return -1;
    }
    public int search(int[] nums, int target) {
        if(nums.length == 0) return -1;
        if(nums.length == 1) return nums[0] == target?0:-1;
        int largeMin = nums[0];
        int smallMax = nums[nums.length - 1];
        int begin = 0;
        int end = nums.length;
        //if target >= largeMin, target in first part, else second pard
        if(target < largeMin && target > smallMax) return -1;
        while(begin < end){
            int mid = begin + (end - begin) / 2;
            if(nums[mid] == target) return mid;
            if(target >= largeMin){
                if(nums[mid] > target)
                    end = mid;
                else{
                    if(nums[mid] <= smallMax)
                        end = mid;
                    else
                        begin = mid + 1;
                }
            }else if(target <= smallMax){
                if(nums[mid] > target){
                    if(nums[mid] >= largeMin )
                        begin = mid + 1;
                    else
                        end = mid;
                }
                else
                    begin = mid + 1;
            }
        }
        return -1;
    }

    public ListNode insertionSortList(ListNode head) {
        if(head == null || head.next == null) return head;
        ListNode dummy = new ListNode(Integer.MIN_VALUE);
        dummy.next = head;
        ListNode tempHead = head;
        ListNode tempEnd = head;
        ListNode cur = tempHead.next;

        while(cur != null){
            ListNode temp = dummy;
            while(temp.next!= cur && !(cur.val <= temp.next.val && cur.val >= temp.val)){
                temp = temp.next;
            }
            if(temp.next!= cur){
                tempEnd.next = cur.next;
                cur.next = temp.next;
                temp.next = cur;
            }
            tempEnd = tempEnd.next;
        }
        return dummy.next;

    }
    class TreeNode{
        int val;
     TreeNode left;
      TreeNode right;
      TreeNode(int x) { val = x; }
    }
    int index = 0;
    public String serialize(TreeNode root) {
        if(root == null) return "#";
        StringBuilder sb = build(root,new StringBuilder(""));
        return sb.toString();
    }
    public StringBuilder build(TreeNode root,StringBuilder sb){
        if(root == null){
            sb.append("#");
            return sb;
        }
        sb.append(Integer.toString(root.val));
        sb.append(',');
        build(root.left,sb);
        sb.append(',');
        build(root.right,sb);
        return sb;
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        String[] list = data.split(",");
        System.out.println(data);
        index = 0;
        return help(list);
    }
    public TreeNode help(String[] data){
        if(index >= data.length || data[index].equals("#")){
            //index++;
            return null;
        }
        TreeNode root = new TreeNode(Integer.valueOf(data[index]));
        index++;
        root.left = help(data);
        index++;
        root.right = help(data);
        return root;
    }
    public int findCircleNum(int[][] M) {
        if(M == null || M.length == 0 || M[0].length == 0) return 0;
        Integer code = 0;
        HashMap<Integer,ArrayList<Integer>> map = new HashMap<Integer,ArrayList<Integer>>();
        for(int i = 0; i < M.length;i++){

            for(int j = 0; j < M[0].length;j++){

                if(i!=j && M[i][j] == 1){
                    if(!map.containsKey(i)){
                        ArrayList<Integer> list = new ArrayList<>();
                        list.add(code++);
                        map.put(i,list);
                    }
                    if(map.containsKey(j)){
                        int temp = map.get(i).get(0);
                        map.get(j).remove(0);
                        map.get(j).add(temp);
                    }else{
                        map.put(j,map.get(i));
                    }
                }
                else if(i == j){
                    if(!map.containsKey(i)){
                        ArrayList<Integer> list = new ArrayList<>();
                        list.add(code++);
                        map.put(i,list);
                    }
                }
            }
        }
        HashSet<Integer> set = new HashSet<Integer>();
        for(Map.Entry<Integer,ArrayList<Integer>> e: map.entrySet()){
            set.add(e.getValue().get(0));
        }
        return set.size();
    }
    HashMap<Character,HashSet<Character>> map = new HashMap<Character,HashSet<Character>>();
    public String alienOrder(String[] words) {
        if(words.length == 0) return "";
        if(words.length == 1) return words[0];

        HashSet<Character> allAppearance = new HashSet<>();
        for(int i = 0; i < words[0].length();i++){
            allAppearance.add(words[0].charAt(i));
        }
        for(int i = 1; i < words.length;i++){
            int leng = Math.min(words[i].length(),words[i - 1].length());
            int j = 0;
            for(; j < leng; j++){
                allAppearance.add(words[i].charAt(j));
                if(words[i].charAt(j)!=words[i - 1].charAt(j)){
                    break;
                }
            }
            for(int k = j + 1; k < words[i].length(); k++){
                allAppearance.add(words[i].charAt(k));
            }
            if(j != leng){
                if(map.containsKey(words[i - 1].charAt(j))){
                    map.get(words[i - 1].charAt(j)).add(words[i].charAt(j));
                }else{
                    HashSet<Character> tempset = new HashSet<>();
                    tempset.add(words[i].charAt(j));
                    map.put(words[i - 1].charAt(j),tempset);
                }
                if(map.containsKey(words[i].charAt(j)) && map.get(words[i].charAt(j)).contains(words[i - 1].charAt(j)))
                    return "";
            }
        }

        HashSet<Character> visited = new HashSet<>();
        Stack<Character> stack = new Stack<>();
        for(Map.Entry<Character,HashSet<Character>> e:map.entrySet()){
            if(visited.contains(e.getKey()))
                continue;
            toposort(stack,visited,e.getKey());

        }
        StringBuffer sb = new StringBuffer("");
        for(int i = 0; i < stack.size();i++){
            sb.append(stack.get(i));
        }
        return sb.toString();

    }
    private void toposort(Stack<Character> stack,HashSet<Character> visited,Character root){
        visited.add(root);
        if(!map.containsKey(root))
            stack.add(root);
        else{
            for(Character c: map.get(root)){
                if(visited.contains(c))
                    continue;
                toposort(stack,visited,c);
            }
            stack.add(root);
        }
    }
    /*
    ArrayList<String> ans = new ArrayList<>();
    public List<String> restoreIpAddresses(String s) {
        if(s == null || s.length() > 12 || s.length() < 4)
            return ans;
        ArrayList<Integer> list = new ArrayList<>();
        list.add(0);
        for(int i = 1; i < 4;i++){
            list.add(i);
            helper(s,list);
            list.remove(1);
        }
        return ans;
    }
    void helper(String s,ArrayList<Integer> list){
        if(list.get(list.size() - 1) >= s.length()){
            //list.remove(list.size() - 1);
            return;
        }
        else{
            String temp = s.substring(list.get(list.size() - 2),list.get(list.size() - 1));
            int last = Integer.valueOf(temp);
            if(last >255 || last < 0 || (temp.charAt(0) == '0' && temp.length() > 1)){
                //list.remove(list.size() - 1);
                return;
            }
            if(list.size() == 4){
                temp = s.substring(list.get(list.size() - 1));
                int end = Integer.valueOf(temp);
                if(end < 256 && end > -1 && !(temp.charAt(0) == '0' && temp.length() > 1)){
                    build(s,list);
                    return;
                }else{
                    //list.remove(3);
                    return;
                }
            }else{
                int index = list.get(list.size() - 1);
                for(int i = index + 1; i < index + 4; i++){
                    list.add(i);
                    helper(s,list);
                    list.remove(list.get(list.size() - 1));
                }
            }
        }
    }
    void build(String s, ArrayList<Integer>  list){
        list.add(s.length());
        StringBuffer sb = new StringBuffer("");
        for(int i = 0; i < 4;i++){
            sb.append(s.substring(list.get(i),list.get(i + 1)));
            sb.append('.');
        }
        sb.deleteCharAt(sb.length() - 1);
        ans.add(sb.toString());
        list.remove(4);
    }*/
    double compute(double x, double guess)
    {

        for(int i = 10; i !=0;i--){

            guess = (3-x*guess*guess)*(x*guess);
        }
        return guess;
        // ...
    }
    double tr(double x,int n){
        double xmm1 = x;
        int eax = 0;
        if(n ==0 )
            return 1;
        x = 1;
        if(n%255 > 1){
            x = xmm1 * x;
            if(n <=1)
                return x;
        }
        while(n%255 <= 1){
            xmm1*=xmm1;
            eax = n;
            eax >>= 31;
            eax += n;
            eax/=2;
            if(n <= 1) return x;
            n = eax;
            if(n%255 <=1)
                continue;
            x *= xmm1;
            if(n <= 1)
                return x;
        }
        return x;

    }
    public int lenLongestFibSubseq(int[] A) {
        if(A == null) return 0;
        if(A.length < 3 )
            return A.length;
        int max = 0;
        //HashMap<Integer,ArrayList<ArrayList<Integer>>> map = new HashMap<>();
        //HashMap<Integer,Set<Integer>> map1 = new HashMap<>();
        ArrayList<HashMap<Integer,Integer>> list = new ArrayList<>();
        //val:list.get(0) is index, list.get(1) is the sequence length
        //HashMap<Integer,Integer> map = new HashMap<Integer,Integer>();
        for(int i = 0; i < A.length;i++){
            list.add(new HashMap<Integer,Integer>());
            for(int j = 0; j < i;j++){
                int sum = A[i] + A[j];
                list.get(i).put(sum,2);
            }
        }
        for(int i = 0; i < A.length;i++){
            for(int j = 0; j < i;j++){
                if(list.get(j).containsKey(A[i])){
                    int newSum = A[i] + A[j];
                    int newLength = list.get(j).get(A[i]) + 1;
                    list.get(i).put(newSum,newLength);
                    //if(newLength > max)
                      //  System.out.println(i);
                    max = newLength;
                }
            }
        }
        return max;

    }
    public static String decode(String encoded) {
        StringBuffer sb = new StringBuffer(encoded);
        sb.reverse();

        StringBuffer ans = new StringBuffer("");

        for(int i = 0;i < encoded.length();){
            if(sb.charAt(i) == '1'){
                int ascii = Integer.parseInt(sb.substring(i,i + 3));
                ans.append(Character.toString((char)ascii));
                i += 3;
            }else{
                int ascii = Integer.parseInt(sb.substring(i,i + 2));
                ans.append(Character.toString((char)ascii));
                i += 2;
            }
        }
        return ans.toString();


    }
    String electionWinner(String[] votes) {
        HashMap<String,Integer> counting = new HashMap<>();
        int max = 0;
        for(String name: votes){
            if(counting.containsKey(name)){
                counting.put(name, counting.get(name) + 1);
            }else{
                counting.put(name, 1);
            }
            if (counting.get(name) > max) {
                max = counting.get(name);
            }
        }
        String elected = "1";
        for(Map.Entry<String,Integer> e: counting.entrySet()){
            if(e.getValue() == max){
                if(e.getKey().compareTo(elected) > 0){
                    elected = e.getKey();
                }
            }
        }
        return elected;


    }




        /*
         * Complete the 'countMatches' function below.
         *
         * The function is expected to return an INTEGER.
         * The function accepts following parameters:
         *  1. STRING_ARRAY grid1
         *  2. STRING_ARRAY grid2
         */

        public  int countMatches(List<String> grid1, List<String> grid2) {
            // Write your code here
            if(grid1.size() == 0 || grid1.get(0).length() == 0) return 0;
            char[][] g1 = new char[grid1.size()][grid1.get(0).length()];
            char[][] g2 = new char[grid2.size()][grid2.get(0).length()];
            for(int i = 0; i < grid1.size();i++){
                for(int j = 0; j < grid1.get(0).length();j++){
                    g1[i][j] = grid1.get(i).charAt(j);
                    g2[i][j] = grid2.get(i).charAt(j);
                }
            }
            int ans = 0;
            int visited = 2;
            for (int i = 0; i < grid1.size(); i++) {
                for (int j = 0; j < grid1.get(0).length(); j++) {
                    if(g1[i][j] == '1'){
                        Map<Integer,Set<Integer>> map = new HashMap<>();
                        erase(g1,i,j,map);
                        if(check(g2,i,j,map,(char)visited)){
                            ans += 1;
                        }
                    }
                }
            }
            return ans;
        }
       public void erase(char[][] grid,int i,int j,Map<Integer,Set<Integer>> map){
            if(i < 0 || j < 0 || i >= grid.length || j >= grid[0].length || grid[i][j] == '0')
                return;
            grid[i][j] = '0';
            if(map.containsKey(i)){
                map.get(i).add(j);
            }else{
                HashSet<Integer> set = new HashSet<>();
                set.add(j);
                map.put(i,set);
            }
            erase(grid, i + 1, j, map);
            erase(grid, i - 1, j, map);
            erase(grid, i, j + 1, map);
            erase(grid, i, j - 1, map);
        }
        public boolean check(char[][] grid,int i,int j,Map<Integer,Set<Integer>> map,char visited){
            if(i < 0 || j < 0 || i >= grid.length || j >= grid[0].length){
                return true;
            }
            if(grid[i][j] == visited) return true;
            if(grid[i][j] == '0'){
                if(map.containsKey(i) && map.get(i).contains(j))
                    return false;
                return true;
            }
            grid[i][j] = visited;
            if(!map.containsKey(i)){
                return false;
            }
            if(!map.get(i).contains(j)){
                return false;
            }
            return  check(grid, i + 1, j, map,visited) && check(grid, i - 1, j, map,visited) && check(grid, i, j + 1, map,visited) & check(grid, i, j - 1, map,visited);


        }

    public int subarraySum(int[] nums, int k) {
        if(nums.length < 1) return 0;
        int[] presum = new int[nums.length + 1];
        presum[0] = 0;
        HashMap<Integer,ArrayList<Integer>> map = new HashMap<>();
        for(int i = 1; i < presum.length;i++){
            presum[i] = presum[i - 1] + nums[i - 1];
            if(map.containsKey(presum[i])){
                map.get(presum[i]).add(i);
            }else{
                ArrayList<Integer> li = new ArrayList<Integer>();
                li.add(presum[i]);
                map.put(presum[i],li);
            }
        }
        int total = 0;
        for(int i = 0; i < presum.length;i++){
            int diff = k + presum[i];
            if(map.containsKey(diff)){
                int index = Collections.binarySearch(map.get(diff),i);
                if(index < 0){
                    total += map.get(diff).size() - (index + 1) * (-1) ;
                }else{
                    total += map.get(diff).size() - index - 1;
                }

            }
        }
        return total;

    }
    void sort(int arr[])
    {
        int n = arr.length;

        // The output character array that will have sorted arr
        int output[] = new int[n];

        // Create a count array to store count of inidividul
        // characters and initialize count array as 0
        int count[] = new int[256];
        for (int i=0; i<256; ++i)
            count[i] = 0;

        // store count of each character
        for (int i=0; i<n; ++i)
            ++count[arr[i]];

        // Change count[i] so that count[i] now contains actual
        // position of this character in output array
        for (int i=1; i<=255; ++i)
            count[i] += count[i-1];

        // Build the output character array
        // To make it stable we are operating in reverse order.
        for (int i = n-1; i>=0; i--)
        {
            output[count[arr[i]]-1] = arr[i];
            --count[arr[i]];
        }

        // Copy the output array to arr, so that arr now
        // contains sorted characters
        for (int i = 0; i<n; ++i)
            arr[i] = output[i];
    }
    public int totalFruit(int[] tree) {

        int sum = 0;
        String last_type = String.valueOf(tree[0]);
        int last_type_index = 0;
        int last_continuous_counter = 1;
        for(int i = 0; i < tree.length;i++){
            String first = last_type;
            String second = "a";
            int first_counter = last_continuous_counter;
            int second_counter = 0;

            boolean end = true;

            for(int j = i + 1; j < tree.length;j++){
                if(first.equals("a")){
                    first = String.valueOf(tree[j]);
                    first_counter += 1;
                }
                else if(first.equals(String.valueOf(tree[j]))){
                    first_counter += 1;
                }
                else if(second.equals("a")){
                    second = String.valueOf(tree[j]);
                    second_counter += 1;
                }else if(second.equals(String.valueOf(tree[j]))){
                    second_counter += 1;
                }else{
                    sum = Math.max(sum,second_counter + first_counter);
                    end = false;
                    i = j - 2;
                    break;
                }
                if(end){
                    if(!String.valueOf(tree[j]).equals(last_type)){
                        last_type = String.valueOf(tree[j]);
                        last_continuous_counter = 1;
                        last_type_index = j;
                    }else{
                        last_continuous_counter += 1;
                    }
                }
            }
            sum = Math.max(sum,second_counter + first_counter);
            if(end)
                return sum;

        }
        return sum;


    }
   void hanoiTower(int from,int to,int intermediate,int level){
            if(level == 1){
                System.out.println("moving plate "+level+" from rod" + Integer.toString(from) +" to "+Integer.toString(to));
                return;
            }
            hanoiTower(from,intermediate,to,level - 1);
            System.out.println("moving plate" +level+"from rod" + Integer.toString(from) +" to "+Integer.toString(to));
            hanoiTower(intermediate,to,from,level - 1);
   }
    public List<String> restoreIpAddresses(String s) {
        List<String> ans = new ArrayList<String>();
        List<String> input = new ArrayList<String>();
        helper(input,s,ans);
        return ans;
    }
    private void helper(List<String> input,String s,List<String> ans){
        if( s.length() < 1 || (s.charAt(0) == 0 && s.length() > 1))
            return;
        if(input.size() == 3){
            int temp = Integer.parseInt(s);
            if( temp >= 0 && temp < 256)
                ans.add(input.get(0)+"."+input.get(1)+"."+input.get(2)+"."+temp);
            System.out.println("here");
            return;

        }
        if(s.length() < 2)
            return;
        for(int i = 1; i < 4 && i <= s.length(); i++){
            List<String> list = new ArrayList<String>(input);
            String str = s.substring(0,i);
            int temp = Integer.parseInt(str);
            if( temp >= 0 && temp < 256){
                list.add(str);
                helper(list,s.substring(i),ans);
            }


        }
    }
    //int index = 0;
    boolean flag = true;
    public int[] findOrder(int numCourses, int[][] prerequisites) {
        HashMap<Integer,List<Integer>> graph = new HashMap<Integer,List<Integer>>();
        for(int i = 0; i < prerequisites.length;i++){
            if(graph.containsKey(prerequisites[i][0])){
                graph.get(prerequisites[i][0]).add(prerequisites[i][1]);
            }
            else{
                ArrayList<Integer> edges = new ArrayList<Integer>();
                edges.add(prerequisites[i][1]);
                graph.put(prerequisites[i][0],edges);
            }
        }
        HashSet<Integer> visited = new HashSet<Integer>();
        HashSet<Integer> learnt = new HashSet<Integer>();
        int[] ans = new int[numCourses];
        for(Integer node:graph.keySet()){
            if(!flag){
                System.out.println("impossible");
                return new int[0];
            }
            if(learnt.contains(node))
                continue;
            visited.add(node);
            helper(visited,learnt,node,graph,ans);
            learnt.add(node);
            ans[this.index] = node;
            this.index++;
        }
        if(!flag)
            return new int[0];
        for(int i = 0; i < numCourses;i++){
            if(learnt.contains(i) == false){
                ans[this.index++] = i;
            }
        }
        return ans;

    }
    private void helper(HashSet<Integer> visited, HashSet<Integer> learnt
            ,int node,HashMap<Integer,List<Integer>> graph,int[] ans ){
        List<Integer> dependencies = graph.get(node);
        for(Integer dependency:dependencies){
            if(learnt.contains(dependency) ){
                continue;
            }
            if(!graph.containsKey(dependency)){
                learnt.add(dependency);
                ans[this.index++] = dependency;
                continue;
            }
            if(!learnt.contains(dependency) && visited.contains(dependency)){
                this.flag = false;
                return;
            }
            visited.add(dependency);
            helper(visited,learnt,dependency,graph,ans);
            learnt.add(dependency);
            ans[this.index] = node;
            this.index++;
        }
    }
    public boolean isBipartite(int[][] graph) {
        HashSet<Integer> set1 = new HashSet<Integer>();
        HashSet<Integer> set2 = new HashSet<Integer>();
        LinkedList<Integer> list = new LinkedList<Integer>();
        for(int i = 0; i < graph.length;i++){
            if(!set1.contains(i) && !set2.contains(i)) {

                if (!set2.contains(i))
                    set2.add(i);
                else
                    set1.add(i);
            }
            if(graph[i].length == 0) continue;
            if(!bfs(set1,set2,i,graph))
                return false;
        }
        return true;
    }
    boolean bfs( HashSet<Integer> set1, HashSet<Integer> set2,int index,int[][] graph){
        LinkedList<Integer> queue = new LinkedList<Integer>();
        HashSet<Integer> insertion_set = null;
        HashSet<Integer> check_set = null;
        if(set1.contains(index)){
            insertion_set = set2;
            check_set = set1;
        }
        else{
            insertion_set = set1;
            check_set = set2;
        }
        for(int i = 0; i < graph[index].length;i++){
            queue.addLast(graph[index][i]);
            if(check_set.contains(graph[index][i]))
                return false;
            insertion_set.add(graph[index][i]);
        }
        HashSet<Integer> temp = check_set;
        check_set = insertion_set;
        insertion_set = temp;
        while(!queue.isEmpty()){
            int size = queue.size();
            for(int i = 0; i < size;i++){
                int cur = queue.removeFirst();
                for(int j = 0; j < graph[cur].length;j++){
                    if(check_set.contains(graph[cur][j]))
                        return false;
                    if(!insertion_set.contains(graph[cur][j])){
                        queue.add(graph[cur][j]);
                        insertion_set.add(graph[cur][j]);
                    }
                }
            }
            temp = check_set;
            check_set = insertion_set;
            insertion_set = temp;
        }
        return true;

    }
    public List<List<Integer>> levelOrderBottom(TreeNode root) {
        List<List<Integer>> ans = new ArrayList<List<Integer>>();
        helper(ans,root,0);
        return ans;
    }
    void helper(List<List<Integer>> ans,TreeNode root,int level){
        if(root == null) return;
        if(level == ans.size()){
            ArrayList<Integer> list = new ArrayList<Integer>();
            list.add(root.val);
            ans.add(list);
        }else{
            ans.get(level).add(root.val);
        }
        helper(ans,root.left,level+1);
        helper(ans,root.right,level+1);

    }


        ArrayList<HashSet<Integer>> findConnection(int[][] graph){
            HashMap<Integer,HashSet<Integer>> map = new HashMap<Integer,HashSet<Integer>>();
            for(int i = 0; i < graph.length;i++){
                if(map.containsKey(graph[i][0])){
                    map.get(graph[i][0]).add(graph[i][0]);
                }else{
                    HashSet<Integer> list = new HashSet<Integer>();
                    list.add(graph[i][1]);
                    map.put(graph[i][0],list);
                }
            }
            ArrayList<HashSet<Integer>> ans = new ArrayList<HashSet<Integer>>();
            HashMap<Integer,HashSet<Integer>> alreadyUsedEdges = new HashMap<Integer,HashSet<Integer>>();
            for(int i = 0; i < graph.length;i++){
                if(alreadyUsedEdges.containsKey(graph[i][0])){
                    if(alreadyUsedEdges.get(graph[i][0]).contains(graph[i][1])){
                        continue;
                    }
                    HashSet<Integer> temp = new HashSet<Integer>();
                    temp.add(graph[i][0]);
                    helper(ans,temp,alreadyUsedEdges,map,graph[i][0]);
                }
            }
            return ans;
        }
        void helper(ArrayList<HashSet<Integer>> ans, HashSet<Integer> part,
                    HashMap<Integer,HashSet<Integer>> alreadyUsedEdges,HashMap<Integer,HashSet<Integer>> graphMap,int node){
        if(alreadyUsedEdges.containsKey(node)){
            part.add(node);
            for(Integer p:part){
                alreadyUsedEdges.get(node).add(p);
            }
            return;
        }
        else{
            if(!graphMap.containsKey(node)){//reached the end
                for(Integer p :part){
                    if(alreadyUsedEdges.containsKey(p)){

                    }
                }
            }
        }


        }

    public int countComponents(int n, int[][] edges) {
        HashMap<Integer,HashSet<Integer>> map = new HashMap<>();
        for(int i = 0; i < edges.length;i++){
            if(!map.containsKey(edges[i][0])){
                HashSet<Integer> tmp = new HashSet<>();
                tmp.add(edges[i][1]);
                map.put(edges[i][0],tmp);
            }else{
                map.get(edges[i][0]).add(edges[i][1]);
            }
            if(!map.containsKey(edges[i][1])){
                HashSet<Integer> tmp = new HashSet<>();
                tmp.add(edges[i][0]);
                map.put(edges[i][1],tmp);
            }else{
                map.get(edges[i][1]).add(edges[i][0]);
            }
        }
        int ans = 0;
        HashSet<Integer> visited = new HashSet<Integer>();
        ArrayList<HashSet<Integer>> agg = new ArrayList<>();
        for(int i = 0; i < edges.length;i++){
            HashSet<Integer> tmp = new HashSet<Integer>();
            if(visited.contains(edges[i][0]))
                continue;
            helper(map,edges[i][0],tmp);
            ans++;
            visited.addAll(tmp);
        }

        return ans;
    }
    void helper(HashMap<Integer,HashSet<Integer>> map,int node,HashSet<Integer> visited){
        if(visited.contains(node))
            return;
        visited.add(node);
        for(Integer link:map.get(node)){
            helper(map,link,visited);
        }

    }
    int find_rotation(int[] arr){
        int left = 0;
        int right = arr.length - 1;
        //int last_move = 1;//1:left to right 2:right to left
        //int last_number = arr[0];
        /*
        while(left!=right){
            int mid = (left + right) / 2;
            if(mid == 0 || mid == arr.length - 1 )
                return mid;
            else if(arr[mid - 1] > arr[mid] && arr[mid + 1] > arr[mid])
                return mid;
            if(arr[mid] > last_number ){
                if(last_move == 2){
                    left = mid + 1;
                    last_move = 1;
                }
                else if(last_move == 1) {
                    left = mid + 1;
                    last_move = 1;
                }
            }if(arr[mid] <= last_number){
                if(last_move == 1){
                    right = mid;
                    last_move = 2;
                }else if(last_move == 2){
                    right = mid
                }
            }

        }
        */

        while(left < right){
            int mid = left + (right - left)/2;
            if(arr[mid] > arr[right]){
                left = mid + 1;
            }
            else if(arr[mid] < arr[right]){
                right = mid;
            }else{
                right--;
            }
        }
        return left;

    }
    public List<List<String>> accountsMerge(List<List<String>> accounts) {
        HashMap<String,Set<String>> graph = new HashMap<String,Set<String>>();
        for(int i = 0; i < accounts.size();i++){
            String name = Integer.toString(i);
            HashSet<String> set1 = new HashSet<String>();
            graph.put(name,set1);

            for(int j = 1; j < accounts.get(i).size();j++){

                String temp = accounts.get(i).get(j);

                if(graph.containsKey(temp)){
                    graph.get(temp).add(name);
                }
                else{
                    HashSet<String> set = new HashSet<String>();
                    set.add(name);
                    graph.put(temp,set);
                }
                set1.add(temp);
            }
        }
        ArrayList<List<String>> ans = new ArrayList<List<String>>();
        HashSet<String> visited = new HashSet<String>();
        for(int i = 0; i < accounts.size();i++){
            HashSet<Integer> set = new HashSet<Integer>();
            if(visited.contains(Integer.toString(i)))
                continue;
            else{
                dfs(visited,graph,Integer.toString(i),set);
            }
            HashSet<String> merged = new HashSet<String>();
            for(Integer index:set){
                for(int j = 1; j < accounts.get(index).size();j++){
                    merged.add(accounts.get(index).get(j));
                }
            }
            LinkedList<String> tmp = new LinkedList<String>();
            tmp.addAll(merged);
            Collections.sort(tmp);
            tmp.addFirst(accounts.get(set.iterator().next()).get(0));
            ans.add(tmp);
        }
        return ans;
    }

    void dfs(HashSet<String> visited,HashMap<String,Set<String>> graph,String node,HashSet<Integer> merge){
        if(visited.contains(node))
            return;
        visited.add(node);
        if(Character.isDigit(node.charAt(0))){
            merge.add(Integer.parseInt(node));
        }
        //System.out.println(node);
        for(String child: graph.get(node)){
            dfs(visited,graph,child,merge);
        }

    }
    /*
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        Collections.sort(candidates);
        for(int i = 0; i < candidates.length;i++){
            dfs(candidates,target,0,new LinkedList<Integer>(),0);
        }
        return ans;

    }
    boolean dfs(int[] candidates,int target,int index,LinkedList<Integer> prefix,int sum){
        if(index >= candidates.length){
            return false;
        }
        if(candidates[index] + sum > target){
            return false;
        }
        sum += candidates[index];
        prefix.add(candidates[index]);
        if(sum == target){
            ans.add(new LinkedList<Integer>(prefix));
        }
        for(int i = index + 1; index < candidates.length;i++){
            if(!dfs(candidates,target,i,prefix,sum))
                break;
        }

        return true;
    }
    */

    public int[] prisonAfterNDays(int[] cells, int N) {
        HashMap<String,Integer> map = new HashMap<>();


        int counter = 0;
        int index = 0;
        ArrayList<String> li = new ArrayList<String>();
        while(true){
            StringBuffer sb = new StringBuffer("");
            for(int i = 0; i < cells.length;i++){
                sb.append((char)(cells[i] + '0'));
            }
            String tmp = sb.toString();
            if(N == counter)
                return cells;
            if(map.containsKey(tmp)){
                //System.out.println("aa");
                index = map.get(tmp);
                break;
            }else{
                map.put(tmp,counter);
                li.add(tmp);
                counter++;
                int[] tmpCells = new int[8];
                tmpCells[0] = 0;
                tmpCells[7] = 0;
                for(int i = 1; i < 7;i++){
                    if(cells[i - 1] == cells[i + 1]){
                        tmpCells[i] = 1;
                    }else{
                        tmpCells[i] = 0;
                    }
                }
                cells = tmpCells;
            }
        }
        int[] ans  = new int[8];
        int cycle_length = counter - index;
        System.out.println(cycle_length);
        String tmp = li.get(index + (N - index)%(cycle_length));
        for(int i = 0; i < tmp.length();i++){
            ans[i] = tmp.charAt(i) - '0';
        }
        for(int c:ans){
            System.out.print(c);
        }
        return ans;

    }

    public int maxSubarraySumCircular(int[] A) {

        int sum = 0;
        int max = A[0];
        int[] max_arr = new int[A.length];
        for(int i = 0; i < A.length;i++){
            if(A[i] > max)
                max = A[i];
        }
        if(max < 0)
            return max;

        for(int i = 0; i < A.length;i++){
            if(A[i] > 0){
                sum += A[i];
                if(sum > max)
                    max = sum;
            }else{
                if(A[i] + sum > 0){
                    sum += A[i];
                    if(sum > max)
                        max = sum;
                }else{
                    sum = 0;
                }
            }
            max_arr[i] = max;
        }



        int forward[] = new int[A.length];
        int back[] = new int[A.length];
        int prev = 0;
        for(int i = 0; i < A.length;i++){
            prev += A[i];
            forward[i] = prev;
        }
        for(int i = 0; i < A.length - 1;i++){
            if(forward[i] > forward[i + 1])
                forward[i + 1] = forward[i];
        }
        int pre = 0;
        for(int i = A.length - 1; i >= 0;i--){
            pre += A[i];
            back[A.length - 1- i] = pre;
        }
        for(int i = 0; i < A.length - 1;i++){
            if(back[i] > back[i + 1])
                back[i + 1] = back[i];
        }
        int second_max = 0;
        for(int i = 0; i < A.length - 1;i++){
            if(second_max < back[i] + forward[A.length -1 - i ])
                second_max = back[i] + forward[A.length -1 - i - 1];
        }
        System.out.println(max);
        System.out.println(second_max);

        return Math.max(max,second_max);
    }
    public List<String> generateAbbreviations(String word) {
        ArrayList<String> li = new ArrayList<>();
        li.add(word);
        for(int j = 0; j < word.length();j++) {
            for (int i = 1; i <= word.length(); i++)
                abbre_helper(li, word, j, i);
        }
        return li;
    }
    void abbre_helper(ArrayList<String> ans,String word,int index,int abb_num){
        if(index + abb_num > word.length())
            return;
        String pattern = word.substring(0,index) + Integer.toString(abb_num) + word.substring(index + abb_num);
        ans.add(pattern);
        //System.out.println(pattern);
        for(int i = 2; i + index < word.length();i++){
            for(int j = 1; j < word.length();j++){
                abbre_helper(ans,pattern,index + i,j);
            }
        }

    }
    public class Node<T>{
        Node<T> next;
        T element;
        public Node(T input){
            element = input;
        }
    }


    public static void main(String args[]){

    }

}




