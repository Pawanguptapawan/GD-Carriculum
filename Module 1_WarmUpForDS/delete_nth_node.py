
# A linked list head is given, and a integer n is given. delete the nth node from the end from the linked list without using any extra space.

# class Node for define one node for linked list with two properties its value and its next pointer.
class Node:
    def __init__(self, val):
        self.val = val
        self.next = None

# calculate the length of the linked list.
def ll_length(head):
    count = 0
    temp = head
    while temp:
        temp = temp.next
        count += 1
    return count


def remove_nth_node(head, n):
    length = ll_length(head)   # length of the lined list.
    from_start = length - n - 1    # nth node from end. and length-n-1 from start.
    temp = head   # temp is iterator.
    while temp and from_start > 0:  # while temp is not null and from_start > 0;
        temp = temp.next
        from_start -= 1   # decrease the from_start.
    temp.next = temp.next.next #delete the nth node by making is next node to next of the prev.
    return head   # return the same modified linked list.