import json
import logging
import re
from datetime import datetime
from typing import TypedDict, Optional, Annotated, List, Dict, Any
from typing_extensions import Literal

from langchain_gigachat import GigaChat
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# === Настройки ===
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Конфигурация
API_KEY = "YOUR API KEY"
TELEGRAM_TOKEN = "YOUR TG TOKEN"

# === Инициализация LLM ===
try:
    llm = GigaChat(credentials=API_KEY, verify_ssl_certs=False, model="GigaChat-2")
    logging.info("GigaChat успешно инициализирован")
except Exception as e:
    logging.error(f"Ошибка инициализации GigaChat: {e}")
    raise


# === Состояние агента ===
class ResumeState(TypedDict):
    messages: Annotated[list, add_messages]
    stage: Literal[
        "start",
        "collecting_profile",
        "collecting_internship",
        "selecting_style",
        "generating",
        "editing",
        "final"
    ]
    user_profile: Dict[str, Any]
    internship_description: str
    style: str  # "официальный", "креативный", "минималистичный"
    language: str  # "ru", "en"
    resume_text: Optional[str]
    cover_letter_text: Optional[str]
    edit_count: int  # Счетчик правок


# === Системные промпты ===
RESUME_PROMPT = SystemMessage(content="""
Ты — эксперт по карьерному консультированию и HR-специалист. Твоя задача — помочь студенту создать профессиональное резюме и мотивационное письмо.

Правила работы:
1. Задавай один вопрос за раз для лучшего понимания
2. Сначала собери базовую информацию: ФИО, образование, навыки, опыт
3. Затем узнай детали о стажировке: компания, программа, требования
4. Спроси о предпочтениях по стилю и языку
5. После сбора данных сгенерируй документы
6. Предложи возможность редактирования
7. Будь дружелюбным и профессиональным

Важно: всегда адаптируй документы под конкретную стажировку и требования.
""")


# === Вспомогательные функции ===
def extract_json_from_text(text: str) -> Dict[str, Any]:
    """Извлекает JSON из текста ответа LLM"""
    try:
        # Ищем JSON в тексте
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            return json.loads(json_str)
    except (json.JSONDecodeError, AttributeError):
        pass

    # Если JSON не найден, пытаемся извлечь данные по ключевым словам
    extracted = {}
    text_lower = text.lower()

    # Извлекаем имя
    name_patterns = [r'имя[:\s]+([^\n,]+)', r'фио[:\s]+([^\n,]+)', r'зовут[:\s]+([^\n,]+)']
    for pattern in name_patterns:
        match = re.search(pattern, text_lower)
        if match:
            extracted['name'] = match.group(1).strip()
            break

    # Извлекаем образование
    edu_patterns = [r'образование[:\s]+([^\n,]+)', r'вуз[:\s]+([^\n,]+)', r'университет[:\s]+([^\n,]+)']
    for pattern in edu_patterns:
        match = re.search(pattern, text_lower)
        if match:
            extracted['education'] = match.group(1).strip()
            break

    # Извлекаем навыки
    skills_patterns = [r'навыки[:\s]+([^\n,]+)', r'умения[:\s]+([^\n,]+)', r'знаю[:\s]+([^\n,]+)']
    for pattern in skills_patterns:
        match = re.search(pattern, text_lower)
        if match:
            extracted['skills'] = match.group(1).strip()
            break

    return extracted


def has_basic_info(profile: Dict[str, Any]) -> bool:
    """Проверяет, есть ли базовая информация в профиле"""
    required_fields = ["name", "education", "skills"]
    return all(profile.get(field) and profile[field].strip() for field in required_fields)


def get_missing_info(profile: Dict[str, Any]) -> List[str]:
    """Возвращает список недостающей информации"""
    missing = []
    if not profile.get("name"):
        missing.append("Ваше ФИО")
    if not profile.get("education"):
        missing.append("Ваше образование (вуз, специальность, год)")
    if not profile.get("skills"):
        missing.append("Ключевые навыки (например: Python, SQL, командная работа)")
    if not profile.get("experience"):
        missing.append("Опыт работы или стажировок (если есть)")
    if not profile.get("projects"):
        missing.append("Проекты или достижения (если есть)")
    return missing


# === Узлы графа ===
def router(state: ResumeState) -> ResumeState:
    """Определяет следующий этап на основе текущего состояния"""
    stage = state["stage"]
    profile = state.get("user_profile", {})

    # Логика определения следующего этапа
    if stage == "start":
        return {"stage": "collecting_profile"}
    elif stage == "collecting_profile":
        if has_basic_info(profile):
            return {"stage": "collecting_internship"}
        else:
            return {"stage": "collecting_profile"}
    elif stage == "collecting_internship":
        if state.get("internship_description"):
            return {"stage": "selecting_style"}
        else:
            return {"stage": "collecting_internship"}
    elif stage == "selecting_style":
        if state.get("style") and state.get("language"):
            return {"stage": "generating"}
        else:
            return {"stage": "selecting_style"}
    elif stage == "generating":
        return {"stage": "editing"}
    elif stage == "editing":
        # Проверяем, есть ли финальное сообщение
        if state.get("edit_count", 0) >= 3:
            return {"stage": "final"}
        else:
            return {"stage": "editing"}
    else:
        return {"stage": stage}


def collect_profile_node(state: ResumeState) -> ResumeState:
    """Собирает информацию о пользователе"""
    if not state["messages"]:
        return {
            "messages": [AIMessage(
                content="Привет! Я помогу создать резюме и мотивационное письмо. Расскажите о себе: ФИО, образование, навыки.")],
            "stage": "collecting_profile"
        }

    last_msg = state["messages"][-1].content
    profile = state.get("user_profile", {})

    # Извлекаем информацию из ответа пользователя
    try:
        extraction_prompt = f"""
        Извлеки информацию о пользователе из его сообщения. Верни JSON с полями:
        - name (ФИО)
        - education (образование: вуз, специальность, год)
        - skills (навыки, через запятую)
        - experience (опыт работы/стажировки)
        - projects (проекты)
        - achievements (достижения)

        Если каких-то данных нет — оставь поле пустым.

        Сообщение пользователя: {last_msg}

        Верни только JSON без дополнительного текста.
        """

        response = llm.invoke([HumanMessage(content=extraction_prompt)])
        new_data = extract_json_from_text(response.content)

        logging.info(f"Извлеченные данные: {new_data}")

        # Обновляем профиль
        for key, value in new_data.items():
            if value and isinstance(value, str) and value.strip():
                profile[key] = value.strip()
            elif value and not isinstance(value, str):
                # Если это словарь (например, education), преобразуем в строку
                if isinstance(value, dict):
                    if key == 'education':
                        vuz = value.get('вуз', value.get('vuz', ''))
                        specialty = value.get('специальность', value.get('specialty', ''))
                        year = value.get('год', value.get('year', ''))
                        profile[key] = f"{vuz} {specialty} {year}".strip()
                    else:
                        profile[key] = str(value)
                else:
                    profile[key] = str(value)

        logging.info(f"Обновленный профиль: {profile}")

    except Exception as e:
        logging.error(f"Ошибка извлечения данных: {e}")

    # Определяем, чего не хватает
    missing = get_missing_info(profile)

    if missing:
        if len(missing) == 1:
            question = f"Пожалуйста, уточните: {missing[0]}."
        else:
            question = f"Пожалуйста, уточните: {', '.join(missing[:-1])} и {missing[-1]}."
    else:
        question = "Отлично! Теперь расскажите о стажировке, на которую вы подаёте заявку: название компании, описание программы, основные требования."

    return {
        "user_profile": profile,
        "messages": [AIMessage(content=question)],
        "stage": "collecting_profile"
    }


def collect_internship_node(state: ResumeState) -> ResumeState:
    """Собирает информацию о стажировке"""
    internship_desc = state["messages"][-1].content

    return {
        "internship_description": internship_desc,
        "messages": [AIMessage(
            content="Спасибо! Теперь выберите стиль оформления документов:\n\n1️⃣ Официальный - строгий, деловой стиль\n2️⃣ Креативный - живой, с элементами личности\n3️⃣ Минималистичный - лаконичный, только суть\n\nНапишите номер или название стиля.")],
        "stage": "selecting_style"
    }


def select_style_node(state: ResumeState) -> ResumeState:
    """Определяет стиль и язык документов"""
    msg = state["messages"][-1].content.lower()

    # Определяем стиль
    style = "официальный"  # по умолчанию
    if any(word in msg for word in ["1", "официальн", "делов", "строг"]):
        style = "официальный"
    elif any(word in msg for word in ["2", "креатив", "жив", "личн"]):
        style = "креативный"
    elif any(word in msg for word in ["3", "миним", "лакон", "суть"]):
        style = "минималистичный"

    # Определяем язык
    language = "ru"  # по умолчанию
    if any(word in msg for word in ["англ", "english", "en", "english"]):
        language = "en"

    return {
        "style": style,
        "language": language,
        "messages": [AIMessage(content="Отлично! Генерирую ваши документы... Это может занять несколько секунд.")],
        "stage": "generating"
    }


def generate_documents_node(state: ResumeState) -> ResumeState:
    """Генерирует резюме и мотивационное письмо"""
    profile = state["user_profile"]
    internship = state["internship_description"]
    style = state["style"]
    language = state["language"]

    logging.info(f"Генерация документов для профиля: {profile}")
    logging.info(f"Стажировка: {internship}")
    logging.info(f"Стиль: {style}, Язык: {language}")

    # Описания стилей
    style_descriptions = {
        "официальный": "профессиональный и формальный стиль с деловой лексикой",
        "креативный": "живой стиль с элементами личности и креативности",
        "минималистичный": "лаконичный стиль без лишних слов, только ключевая информация"
    }

    lang_descriptions = {
        "ru": "на русском языке",
        "en": "in English"
    }

    style_desc = style_descriptions.get(style, "профессиональный")
    lang_desc = lang_descriptions.get(language, "на русском языке")

    try:
        prompt = f"""
        Создай два документа {lang_desc} в {style_desc}:

        1. РЕЗЮМЕ (CV/Resume)
        2. МОТИВАЦИОННОЕ ПИСЬМО (Cover Letter)

        ВАЖНО: Используй ТОЛЬКО информацию, предоставленную пользователем ниже. НЕ добавляй вымышленные данные!

        Информация о кандидате:
        - Имя: {profile.get('name', 'Не указано')}
        - Образование: {profile.get('education', 'Не указано')}
        - Навыки: {profile.get('skills', 'Не указано')}
        - Опыт: {profile.get('experience', 'Не указано')}
        - Проекты: {profile.get('projects', 'Не указано')}
        - Достижения: {profile.get('achievements', 'Не указано')}

        Описание стажировки:
        {internship}

        Требования к резюме:
        - Используй ТОЛЬКО данные выше
        - Структура: Контакты, Образование, Навыки, Опыт, Проекты, Достижения
        - Адаптируй под требования стажировки
        - Используй ключевые слова из описания стажировки
        - НЕ добавляй вымышленные данные

        Требования к мотивационному письму:
        - 3-4 абзаца
        - Используй ТОЛЬКО данные кандидата выше
        - Почему кандидат подходит для этой стажировки
        - Почему интересуется компанией/программой
        - Что может привнести в команду
        - Связь между опытом и требованиями

        Формат вывода:
        РЕЗЮМЕ:
        [содержимое резюме]

        МОТИВАЦИОННОЕ ПИСЬМО:
        [содержимое письма]

        Не используй markdown разметку.
        """

        response = llm.invoke([HumanMessage(content=prompt)])
        full_text = response.content

        logging.info(f"Получен ответ от LLM: {full_text[:200]}...")

        # Разделяем на резюме и письмо
        resume_text = ""
        cover_letter_text = ""

        if "МОТИВАЦИОННОЕ ПИСЬМО" in full_text:
            parts = full_text.split("МОТИВАЦИОННОЕ ПИСЬМО", 1)
            resume_text = parts[0].replace("РЕЗЮМЕ:", "").strip()
            cover_letter_text = "МОТИВАЦИОННОЕ ПИСЬМО:" + (parts[1] if len(parts) > 1 else "")
        else:
            # Если разделитель не найден, пытаемся разделить по-другому
            if "РЕЗЮМЕ:" in full_text:
                parts = full_text.split("РЕЗЮМЕ:", 1)
                if len(parts) > 1:
                    resume_text = parts[1]
            else:
                resume_text = full_text

        logging.info(f"Сгенерировано резюме: {resume_text[:100]}...")
        logging.info(f"Сгенерировано письмо: {cover_letter_text[:100]}...")

        return {
            "resume_text": resume_text,
            "cover_letter_text": cover_letter_text,
            "edit_count": 0,
            "messages": [AIMessage(
                content=f"🎉 Ваши документы готовы!\n\n📄 РЕЗЮМЕ:\n\n{resume_text}\n\n📝 МОТИВАЦИОННОЕ ПИСЬМО:\n\n{cover_letter_text}\n\n💡 Хотите что-то изменить? Напишите, что нужно поправить (например: «Сделай резюме короче» или «Перепиши письмо в более официальном тоне»).")],
            "stage": "editing"
        }

    except Exception as e:
        logging.error(f"Ошибка генерации документов: {e}")
        return {
            "messages": [AIMessage(content="Извините, произошла ошибка при генерации документов. Попробуйте еще раз.")],
            "stage": "generating"
        }


def edit_documents_node(state: ResumeState) -> ResumeState:
    """Обрабатывает правки документов"""
    feedback = state["messages"][-1].content
    current_resume = state.get("resume_text", "")
    current_cover = state.get("cover_letter_text", "")
    profile = state["user_profile"]
    internship = state["internship_description"]
    style = state["style"]
    language = state["language"]
    edit_count = state.get("edit_count", 0)

    # Ограничиваем количество правок
    if edit_count >= 3:
        return {
            "messages": [AIMessage(
                content="Вы уже внесли максимальное количество правок (3). Документы готовы к использованию! 🚀\n\nДля создания новых документов используйте команду /start")],
            "stage": "final"
        }

    try:
        edit_prompt = f"""
        Пользователь просит внести правки в документы. Внеси изменения согласно запросу.

        Текущее резюме:
        {current_resume}

        Текущее мотивационное письмо:
        {current_cover}

        Запрос на правку:
        {feedback}

        Требования:
        - Обнови только ту часть, которую просят изменить
        - Сохрани общий стиль ({style}) и язык ({language})
        - Адаптируй под требования стажировки: {internship}
        - Верни оба документа в том же формате

        Формат вывода:
        РЕЗЮМЕ:
        [обновленное резюме]

        МОТИВАЦИОННОЕ ПИСЬМО:
        [обновленное письмо]
        """

        response = llm.invoke([HumanMessage(content=edit_prompt)])
        full_text = response.content

        # Разделяем обновленные документы
        resume_text = current_resume
        cover_letter_text = current_cover

        if "МОТИВАЦИОННОЕ ПИСЬМО" in full_text:
            parts = full_text.split("МОТИВАЦИОННОЕ ПИСЬМО", 1)
            resume_text = parts[0].replace("РЕЗЮМЕ:", "").strip()
            cover_letter_text = "МОТИВАЦИОННОЕ ПИСЬМО:" + (parts[1] if len(parts) > 1 else "")

        return {
            "resume_text": resume_text,
            "cover_letter_text": cover_letter_text,
            "edit_count": edit_count + 1,
            "messages": [AIMessage(
                content=f"✅ Документы обновлены!\n\n📄 ОБНОВЛЕННОЕ РЕЗЮМЕ:\n\n{resume_text}\n\n📝 ОБНОВЛЕННОЕ МОТИВАЦИОННОЕ ПИСЬМО:\n\n{cover_letter_text}\n\n💡 Можно внести еще правки или завершить работу.")],
            "stage": "editing"
        }

    except Exception as e:
        logging.error(f"Ошибка редактирования: {e}")
        return {
            "messages": [AIMessage(content="Извините, произошла ошибка при редактировании. Попробуйте еще раз.")],
            "stage": "editing"
        }


def final_node(state: ResumeState) -> ResumeState:
    """Завершающий узел"""
    return {
        "messages": [AIMessage(
            content="🎉 Спасибо за работу! Ваши документы готовы к отправке. Удачи на стажировке! 🚀\n\nДля создания новых документов используйте команду /start")],
        "stage": "final"
    }


# === Создание графа ===
def create_resume_agent():
    """Создает и компилирует граф агента"""
    workflow = StateGraph(ResumeState)

    # Добавляем узлы
    workflow.add_node("router", router)
    workflow.add_node("collect_profile", collect_profile_node)
    workflow.add_node("collect_internship", collect_internship_node)
    workflow.add_node("select_style", select_style_node)
    workflow.add_node("generate", generate_documents_node)
    workflow.add_node("edit", edit_documents_node)
    workflow.add_node("final", final_node)

    # Устанавливаем точку входа
    workflow.set_entry_point("router")

    # Добавляем условные переходы
    workflow.add_conditional_edges(
        "router",
        lambda state: state["stage"],
        {
            "collecting_profile": "collect_profile",
            "collecting_internship": "collect_internship",
            "selecting_style": "select_style",
            "generating": "generate",
            "editing": "edit",
            "final": "final"
        }
    )

    # Добавляем переходы - каждый узел завершает выполнение
    workflow.add_edge("collect_profile", END)
    workflow.add_edge("collect_internship", END)
    workflow.add_edge("select_style", END)
    workflow.add_edge("generate", END)
    workflow.add_edge("edit", END)
    workflow.add_edge("final", END)

    return workflow.compile()


# === Telegram Bot ===
class ResumeBot:
    def __init__(self):
        self.agent = create_resume_agent()
        self.app = Application.builder().token(TELEGRAM_TOKEN).build()
        self.setup_handlers()

    def setup_handlers(self):
        """Настраивает обработчики команд"""
        self.app.add_handler(CommandHandler("start", self.start_command))
        self.app.add_handler(CommandHandler("help", self.help_command))
        self.app.add_handler(CommandHandler("stop", self.stop_command))
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /start"""
        initial_state = {
            "messages": [AIMessage(
                content="👋 Привет! Я помогу вам создать идеальное резюме и мотивационное письмо для стажировки.\n\n📝 Для начала расскажите о себе: ФИО, вуз, специальность, ключевые навыки.")],
            "stage": "start",
            "user_profile": {},
            "internship_description": "",
            "style": "",
            "language": "ru",
            "resume_text": None,
            "cover_letter_text": None,
            "edit_count": 0
        }
        context.user_data["state"] = initial_state
        await update.message.reply_text(initial_state["messages"][0].content)

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /help"""
        help_text = """
🤖 Помощник по созданию резюме и мотивационных писем

📋 Что я умею:
• Создавать персонализированные резюме
• Писать мотивационные письма
• Адаптировать документы под конкретную стажировку
• Поддерживать разные стили (официальный, креативный, минималистичный)
• Работать на русском и английском языках
• Вносить правки по вашим замечаниям

🚀 Команды:
/start - начать создание документов
/stop - завершить текущий диалог
/help - показать эту справку

💡 Советы:
• Будьте конкретны при описании опыта
• Указывайте релевантные навыки
• Подробно опишите стажировку
• Не стесняйтесь просить правки
        """
        await update.message.reply_text(help_text)

    async def stop_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /stop"""
        context.user_data["state"] = {
            "messages": [AIMessage(content="Диалог завершен. Используйте /start для создания новых документов.")],
            "stage": "final"
        }
        await update.message.reply_text("🛑 Диалог завершен. Используйте /start для создания новых документов.")

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик текстовых сообщений"""
        user_input = update.message.text
        current_state = context.user_data.get("state")

        logging.info(f"Получено сообщение: {user_input}")
        logging.info(f"Текущая стадия: {current_state.get('stage') if current_state else 'None'}")

        if current_state is None:
            await self.start_command(update, context)
            current_state = context.user_data["state"]

        # Проверяем, не завершен ли диалог
        if current_state.get("stage") == "final":
            await update.message.reply_text("Диалог завершен. Используйте /start для создания новых документов.")
            return

        # Добавляем сообщение пользователя
        current_state["messages"].append(HumanMessage(content=user_input))

        try:
            # Запускаем агента
            result = self.agent.invoke(current_state)
            context.user_data["state"] = result

            logging.info(f"Новая стадия: {result.get('stage')}")
            logging.info(f"Количество сообщений: {len(result.get('messages', []))}")

            # Отправляем ответ
            if result["messages"]:
                response = result["messages"][-1].content

                # Разбиваем длинные сообщения
                if len(response) > 4096:
                    chunks = [response[i:i + 4096] for i in range(0, len(response), 4096)]
                    for chunk in chunks:
                        await update.message.reply_text(chunk)
                else:
                    await update.message.reply_text(response)
            else:
                await update.message.reply_text("Произошла ошибка при обработке сообщения. Попробуйте еще раз.")

        except Exception as e:
            logging.error(f"Ошибка обработки сообщения: {e}")
            await update.message.reply_text(
                "Извините, произошла ошибка. Попробуйте еще раз или используйте /start для начала заново.")

    def run(self):
        """Запускает бота"""
        logging.info("🤖 Resume Agent запущен и готов к работе!")
        self.app.run_polling()


# === Запуск приложения ===
if __name__ == "__main__":
    try:
        bot = ResumeBot()
        bot.run()
    except Exception as e:
        logging.error(f"Критическая ошибка: {e}")
        print(f"Ошибка запуска: {e}")
