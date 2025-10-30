from langchain_gigachat import GigaChat
from langchain_core.messages import SystemMessage, HumanMessage

API_KEY = "MDE5YTAyNDAtZGNlYi03MDQyLTkzOTItNmY0NGUwZTQ3ZjkyOjdiOWU2MGY3LWZlYjgtNGMxZi1hZDVkLTI4MmUzODQzODYzNQ=="

llm = GigaChat(credentials=API_KEY, verify_ssl_certs=False)

system_prompt = SystemMessage(content="""
        Ты - помощник для классификации учебных материалов по предметам.
Определи, для какого из следующих предметов материал будет наиболее полезен:

1. Численные методы
2. Компьютерные сети
3. Программирование на python
4. Физика

Проанализируй содержание материала и верни ТОЛЬКО название подходящего предмета без дополнительных объяснений, комментариев или пунктуации.
Если материал не подходит ни к одному предмету, верни "Другой предмет". Если данная ссылка находится в приведенном списке, выведи название раздела, под которым она находится:
Численные методы
https://books.altspu.ru/document/65
https://openedu.ru/course/spbstu/NUMMETH/
http://wiki.cs.hse.ru/%D0%A7%D0%B8%D1%81%D0%BB%D0%B5%D0%BD%D0%BD%D1%8B%D0%B5_%D0%9C%D0%B5%D1%82%D0%BE%D0%B4%D1%8B_2021
https://www.hse.ru/edu/courses/339562855
https://teach-in.ru/course/numerical-methods-part-1
https://www.matburo.ru/st_subject.php?p=dr&rut=d992e77c9b77270bef82d706c585bfda4bdda23e35a9fb73a75809a9bc7c9608

Компьютерные сети
https://proglib.io/p/network-books
https://asozykin.ru/courses/networks_online
https://sites.google.com/view/malikov-m-v/%D1%81%D1%82%D1%83%D0%B4%D0%B5%D0%BD%D1%82%D0%B0%D0%BC/3-%D0%BA%D1%83%D1%80%D1%81/%D0%BA%D0%BE%D0%BC%D0%BF%D1%8C%D1%8E%D1%82%D0%B5%D1%80%D0%BD%D1%8B%D0%B5-%D1%81%D0%B5%D1%82%D0%B8
https://www.journal-altspu.ru/document/129
https://ru.hexlet.io/blog/posts/kompyuternaya-set-chto-eto-takoe-osnovnye-printsipy
https://gb.ru/courses/3731


Программирование на python

https://www.knorus.ru/catalog/informatika/698633-programmnaya-inzheneriya-bakalavriat-magistratura-uchebnik/
https://stepik.org/course/67/promo
https://ru.pythontutor.ru/problem/old/1
https://selectel.ru/blog/courses/course-python/
https://devpractice.ru/python/

Физика
https://madi.ru/438-kafedra-fizika-uchebnye-posobiya-po-lekcionnomu-kursu.html
https://znanierussia.ru/articles/%D0%9A%D0%BB%D0%B0%D1%81%D1%81%D0%B8%D1%87%D0%B5%D1%81%D0%BA%D0%B0%D1%8F_%D0%BC%D0%B5%D1%85%D0%B0%D0%BD%D0%B8%D0%BA%D0%B0
https://bigenc.ru/l/nachala-termodinamiki-7415b1
https://naked-science.ru/tags/elektrodinamika
https://nonfiction.ru/stream/kvantovaya-fizika-za-5-minut-glavnyie-voprosyi-i-idei

если ссылки в этом списке нет, открой ее и проанализируй содержимое самостоятельно

Примеры правильных ответов:
Численные методы
Компьютерные сети
Программирование на python
Физика
Другой предмет
    """)

if __name__ == "__main__":
    while True:
        user_input = input("Введите промпт: ")
        if user_input == "exit":
            break
        user_prompt = HumanMessage(content=user_input)
        messages = [system_prompt, user_prompt]
        response = llm.invoke(messages)
        print(response.content)

