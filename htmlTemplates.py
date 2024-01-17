css = '''
<style>
body {
    margin-bottom: 200px;
}
.chat-message {
    padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex;
}

.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
</style>
'''


bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://cdn-icons-png.flaticon.com/512/12224/12224084.png">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">  
    <div class="message">{{MSG}}</div>
        <div class="avatar">
        <img src="https://cdn-icons-png.flaticon.com/512/3237/3237472.png">
    </div>  
</div>
'''