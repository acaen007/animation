"""
AlphaHold'em Manim Animation
----------------------------
This script creates a polished animation of Figures 2, 3, and 4 from the AlphaHold'em paper (AAAI 2022).
It visualizes the architecture and training flow of the model using Manim Community Edition.

Author: Manus AI
Date: May 24, 2025
"""

from manim import *

import numpy as np

# Set consistent color scheme for the animation
COLOR_SCHEME = {
    "card_tensor": "#3498db",  # Blue
    "action_tensor": "#e74c3c",  # Red
    "card_convnet": "#2980b9",  # Darker blue
    "action_convnet": "#c0392b",  # Darker red
    "embedding": "#9b59b6",  # Purple
    "fc_layers": "#8e44ad",  # Darker purple
    "actor_head": "#f39c12",  # Orange
    "critic_head": "#d35400",  # Darker orange
    "environment": "#27ae60",  # Green
    "replay_buffer": "#16a085",  # Teal
    "loss_action": "#f1c40f",  # Yellow
    "loss_value": "#f39c12",  # Orange
    "gradient": "#e67e22",  # Darker orange
    "background": "#141c24",  # Dark blue/gray
    "text": "#ecf0f1",  # Light gray
}

# Configuration constants
TENSOR_OPACITY = 0.5
ARROW_STROKE_WIDTH = 4
ANIMATION_RUN_TIME = 1.0  # Default animation duration
TEXT_SCALE = 0.6  # Scale for text elements

# Helper functions moved outside of class for global access
def create_tensor_3d(dimensions, color, opacity=TENSOR_OPACITY, include_grid=True):
    h, w, d = dimensions
    # unit cube scaled uniformly, then stretched non-uniformly
    cuboid = Cube()
    cuboid.scale_to_fit_height(h)
    # non-uniform stretch to fit width and depth
    cuboid.stretch(w / cuboid.width, 0)   # X-axis
    cuboid.stretch(d / cuboid.depth, 2)   # Z-axis
    cuboid.set_fill(color, opacity)
    cuboid.set_stroke(color, width=1)

    if include_grid:
        grid = VGroup()
        # front face grid
        for i in range(int(h)+1):
            grid.add(Line([-w/2, -h/2 + i, d/2], [w/2, -h/2 + i, d/2], stroke_width=0.5, color=WHITE))
        for i in range(int(w)+1):
            grid.add(Line([-w/2 + i, -h/2, d/2], [-w/2 + i, h/2, d/2], stroke_width=0.5, color=WHITE))
        # side face grid
        for i in range(int(h)+1):
            grid.add(Line([w/2, -h/2 + i, -d/2], [w/2, -h/2 + i, d/2], stroke_width=0.5, color=WHITE))
        for i in range(int(d)+1):
            grid.add(Line([w/2, -h/2, -d/2 + i], [w/2, h/2, -d/2 + i], stroke_width=0.5, color=WHITE))
        # top face grid
        for i in range(int(w)+1):
            grid.add(Line([-w/2 + i, h/2, -d/2], [-w/2 + i, h/2, d/2], stroke_width=0.5, color=WHITE))
        for i in range(int(d)+1):
            grid.add(Line([-w/2, h/2, -d/2 + i], [w/2, h/2, -d/2 + i], stroke_width=0.5, color=WHITE))
        grid.set_stroke(opacity=0.3)
        return VGroup(cuboid, grid)
    return VGroup(cuboid)



def create_network_block(width, height, depth, name, color):
    block = create_tensor_3d((height, width, depth), color, opacity=0.7, include_grid=False)
    label = Text(name, font_size=24).scale(TEXT_SCALE)
    label.set_color(COLOR_SCHEME["text"])
    label.next_to(block, DOWN, buff=0.2)
    return VGroup(block, label)


def create_flow_arrow(start, end, color=WHITE, buff=0.1):
    if isinstance(start, VMobject):
        sp = start.get_center() + RIGHT * (start.width/2 + buff)
    else:
        sp = start
    if isinstance(end, VMobject):
        ep = end.get_center() + LEFT * (end.width/2 + buff)
    else:
        ep = end
    return Arrow(sp, ep, stroke_width=ARROW_STROKE_WIDTH, color=color)


def create_signal_flow(arrow, n_pulses=3, run_time=2):
    dots = VGroup()
    for i in range(n_pulses):
        dot = Dot(color=WHITE, radius=0.05)
        dot.move_to(arrow.get_start())
        dots.add(dot)
    anims = []
    for i, dot in enumerate(dots):
        delay = i * run_time / n_pulses
        anims.append(Succession(Wait(delay), MoveAlongPath(dot, arrow, run_time=run_time-delay), FadeOut(dot, run_time=0.2)))
    return AnimationGroup(*anims)

def create_equation(tex_string, color=WHITE, scale=1.0):
    """
    Create a mathematical equation using MathTex.
    
    Args:
        tex_string (str): LaTeX string for the equation
        color (str): Color of the equation
        scale (float): Scale factor for the equation
        
    Returns:
        MathTex: Equation object
    """
    equation = MathTex(tex_string, color=color).scale(scale)
    return equation


class PokerFullGameScene(ThreeDScene):
    """Simulate a full Hold'em hand, mapping hero cards & actions into the input tensors."""

    def setup(self):
        self.camera.background_color = COLOR_SCHEME["background"]
        # Zoomed‐out so table at left and tensors at right fit comfortably
        self.set_camera_orientation(phi=0 * DEGREES, theta=-90 * DEGREES, zoom=0.7)

    def construct(self):
        # 1) Table further left
        table = Circle(radius=3, color=COLOR_SCHEME["environment"], fill_opacity=0.3)
        table.shift(LEFT * 6)
        self.play(FadeIn(table))
        self.wait(0.3)

        table_center = table.get_center()

        # 2) Prompt
        prompt = Text("Dealing Texas Hold'em", font_size=36, color=COLOR_SCHEME["text"])
        prompt.next_to(table, UP, buff=0.5)
        self.play(Write(prompt))
        self.wait(0.7)

        # 3) Hero & villain card positions—hero lower, villain higher
        hero_pos = [table_center + np.array([-0.5, -2.0, 0]),
                    table_center + np.array([ 0.5, -2.0, 0])]
        opp_pos  = [table_center + np.array([-0.5,  2.0, 0]),
                    table_center + np.array([ 0.5,  2.0, 0])]

        # 4) Create hero hole cards (face up)
        hero_hole  = [("A","♠"), ("K","♥")]
        hero_cards = VGroup()
        for pos,(r,s) in zip(hero_pos, hero_hole):
            newColor = BLACK
            if s in ["♥", "♦"]:
                newColor = RED
            card = Square(0.6, fill_color=WHITE, fill_opacity=1).move_to(pos)
            lbl  = Text(f"{r}{s}", font_size=24, color=newColor).move_to(card)
            hero_cards.add(VGroup(card,lbl))

        # 5) Opponent hole cards (face-down backs)
        opp_cards = VGroup()
        for pos in opp_pos:
            back = Square(0.6, fill_color=GREY, fill_opacity=1).move_to(pos)
            opp_cards.add(back)

        # 6) Deal animation
        for c in opp_cards + hero_cards:
            self.add(c.shift(UP*5))
            self.play(c.animate.shift(DOWN*5), run_time=0.4)
            self.wait(0.1)
        self.wait(0.3)
        self.play(FadeOut(prompt))

        # 7) Pre-flop bets positioned under hero / above villain
        hero_bet     = Text("Preflop: Hero (SB) calls 10", font_size=24, color=COLOR_SCHEME["text"])\
                          .next_to(hero_cards, DOWN, buff=0.5)
        villain_bet  = Text("Preflop: BB checks",        font_size=24, color=COLOR_SCHEME["text"])\
                          .next_to(opp_cards,   UP,   buff=0.5)
        self.play(Write(hero_bet))
        self.wait(0.3)
        self.play(Write(villain_bet))
        self.wait(0.3)

        # 8) Bigger, split-off tensors on the right
        card_tensor   = create_tensor_3d((1.5, 0.9, 1.8), COLOR_SCHEME["card_tensor"])
        action_tensor = create_tensor_3d((1.5, 0.9, 1.8), COLOR_SCHEME["action_tensor"])
        card_label    = Text("Card Tensor",   font_size=24, color=COLOR_SCHEME["text"])
        action_label  = Text("Action Tensor", font_size=24, color=COLOR_SCHEME["text"])

        # scale up for readability
        card_tensor.scale(1.5)
        action_tensor.scale(1.5)

        card_label.next_to(card_tensor, UP, buff=0.3)
        action_label.next_to(action_tensor, UP, buff=0.3)

        card_tensor.to_corner(UR, buff=1.0)
        card_label.next_to(card_tensor, UP, buff=0.3)
        action_tensor.next_to(card_tensor, DOWN, buff=1.2)
        action_label.next_to(action_tensor, UP, buff=0.3)

        self.play(
            FadeIn(card_tensor), FadeIn(card_label),
            FadeIn(action_tensor), FadeIn(action_label),
        )
        self.wait(0.5)

        # Utility: cell position in a tensor block
        def tensor_pos(tensor, row, col, dep):
            h, w, d = 1.5*1.2, 0.9*1.2, 1.8*1.2
            base = tensor.get_center() + np.array([-w/2, h/2, d/2])
            return base + np.array([col*(w/4), -row*(h/4), -dep*(d/6)])

        flashes = VGroup()

        # 9) Map hero hole → card_tensor at (0,0,0),(0,1,0)
        indices = [(0,0,0),(0,1,0)]
        for card,(r,c,dep) in zip(hero_cards, indices):
            dot   = Dot(radius=0.08, color=YELLOW)
            start = card.get_center()
            end   = tensor_pos(card_tensor, r,c,dep)
            self.add(dot)
            self.play(MoveAlongPath(dot, Line(start, end), run_time=1.0))
            flashes.add(dot)
        self.wait(0.5)

        # 10) Map pre-flop actions → action_tensor depth=0
        for txt,(r,c) in [(hero_bet,(0,1)), (villain_bet,(1,0))]:
            dot   = Dot(radius=0.08, color=RED)
            start = txt.get_center()
            end   = tensor_pos(action_tensor, r,c,0)
            self.add(dot)
            self.play(MoveAlongPath(dot, Line(start, end), run_time=0.8))
            flashes.add(dot)
        self.wait(0.5)

        # 11) Flop reveal centered *on the table*, with no overlap
        flop_vals = [("5","♣"),("J","♦"),("10","♥")]
        flop_cards = VGroup()
        # compute 3 evenly spaced x positions across the table diameter
        xs = np.linspace(table_center[0]-2, table_center[0], 3)
        for x,(r,s) in zip(xs, flop_vals):
            pos = np.array([x, table_center[1], 0])
            c   = Square(0.8, fill_color=WHITE, fill_opacity=1).move_to(pos)

            #color is red for hearts and diamonds, black for clubs and spades
            newColor = BLACK
            if s in ["♥", "♦"]:
                newColor = RED
            lbl = Text(f"{r}{s}", font_size=28, color=newColor).move_to(c)
            flop_cards.add(VGroup(c, lbl))

        for fc in flop_cards:
            self.add(fc.shift(UP*5))
            self.play(fc.animate.shift(DOWN*5), run_time=0.5)
        self.wait(0.5)

        # 12) Map flop → card_tensor depths=1–3
        for card,dep in zip(flop_cards, [1,2,3]):
            dot   = Dot(radius=0.08, color=YELLOW)
            start = card.get_center()
            end   = tensor_pos(card_tensor, 1, dep-1, dep)
            self.add(dot)
            self.play(MoveAlongPath(dot, Line(start, end), run_time=0.7))
            flashes.add(dot)
        self.wait(0.5)

        # 13) Flop bets under hero / above villain
        flop_hero_bet    = Text("Flop: Hero bets 20", font_size=24, color=COLOR_SCHEME["text"])\
                               .next_to(hero_cards, DOWN, buff=1)
        flop_villain_bet = Text("Flop: BB calls 20", font_size=24, color=COLOR_SCHEME["text"])\
                               .next_to(opp_cards,   UP,   buff=1)
        
        self.play(Write(flop_hero_bet))
        self.wait(0.3)
        self.play(Write(flop_villain_bet))
        self.wait(0.3)
        
        for txt,(r,c) in [(flop_hero_bet,(0,2)), (flop_villain_bet,(1,2))]:
            dot   = Dot(radius=0.08, color=RED)
            start = txt.get_center()
            end   = tensor_pos(action_tensor, r,c,1)
            self.add(dot)
            self.play(MoveAlongPath(dot, Line(start, end), run_time=0.8))
        self.wait(0.5)

        # 14) Turn reveal at 4th community slot → depth=4

        card = Square(0.8, fill_color=WHITE, fill_opacity=1)
        label = Text("Q♠", font_size=28, color=BLACK).move_to(card.get_center())
        turn = VGroup(card, label)
        # Position it relative to the shifted table center
        turn.move_to([table_center[0] + 1, table_center[1], 0])

        # Deal animation
        turn.shift(UP * 5)
        self.add(turn)
        self.play(turn.animate.shift(DOWN * 5), run_time=0.5)

        # Map into the tensor
        dot = Dot(radius=0.08, color=YELLOW)
        self.add(dot)
        self.play(
            MoveAlongPath(
                dot,
                Line(
                    turn.get_center(),
                    tensor_pos(card_tensor, 2, 0, 4)
                ),
                run_time=1.0
            )
        )
        self.wait(0.5)
        turn_hero_bet    = Text("Turn: Hero bets 200", font_size=24, color=COLOR_SCHEME["text"])\
                               .next_to(hero_cards, DOWN, buff=1.4)
        turn_villain_bet = Text("Turn: BB calls 200", font_size=24, color=COLOR_SCHEME["text"])\
                               .next_to(opp_cards,   UP,   buff=1.4)
        
        self.play(Write(turn_hero_bet))
        self.wait(0.3)
        self.play(Write(turn_villain_bet))
        self.wait(0.3)

        for txt,(r,c) in [(turn_hero_bet,(0,2)), (turn_villain_bet,(1,2))]:
            dot   = Dot(radius=0.08, color=RED)
            start = txt.get_center()
            end   = tensor_pos(action_tensor, r,c,2)
            self.add(dot)
            self.play(MoveAlongPath(dot, Line(start, end), run_time=0.8))
        self.wait(0.5)

        # 15) River reveal at 5th community slot → depth=5
        card = Square(0.8, fill_color=WHITE, fill_opacity=1)
        label = Text("7♦", font_size=28, color=RED).move_to(card.get_center())
        river = VGroup(card, label)
        river.move_to([table_center[0] + 2, table_center[1], 0])

        # Deal animation
        river.shift(UP * 5)
        self.add(river)
        self.play(river.animate.shift(DOWN * 5), run_time=0.5)

        # Map into the tensor
        dot = Dot(radius=0.08, color=YELLOW)
        self.add(dot)
        self.play(
            MoveAlongPath(
                dot,
                Line(
                    river.get_center(),
                    tensor_pos(card_tensor, 3, 0, 5)
                ),
                run_time=1.0
            )
        )
        self.wait(0.5)

        river_hero_bet    = Text("River: Hero bets All-In", font_size=24, color=COLOR_SCHEME["text"])\
                               .next_to(hero_cards, DOWN, buff=1.8)
        river_villain_bet = Text("River: BB Folds", font_size=24, color=COLOR_SCHEME["text"])\
                               .next_to(opp_cards,   UP,   buff=1.8)
        
        self.play(Write(turn_hero_bet))
        self.wait(0.3)
        self.play(Write(river_villain_bet))
        self.wait(0.3)

        for txt,(r,c) in [(turn_hero_bet,(0,2)), (river_villain_bet,(1,2))]:
            dot   = Dot(radius=0.08, color=RED)
            start = txt.get_center()
            end   = tensor_pos(action_tensor, r,c,2)
            self.add(dot)
            self.play(MoveAlongPath(dot, Line(start, end), run_time=0.8))
        self.wait(0.5)

        # 16) Showdown: flip opp hole cards
        anims = []
        for idx,(r,s) in enumerate([("9","♣"),("J","♣")]):
            anims += [FadeOut(opp_cards[idx])]
            face = Text(f"{r}{s}", font_size=24, color=BLACK).move_to(opp_cards[idx])
            anims += [FadeIn(face)]
        self.play(*anims)
        self.wait(1)

        # 17) Wrap up
        self.play(FadeOut(VGroup(
            hero_cards, flop_cards, turn, river,
            hero_bet, villain_bet, flop_hero_bet, flop_villain_bet
        )))
        note = Text("→ These tensors feed into the next scene",
                    font_size=28, color=COLOR_SCHEME["text"]).to_edge(DOWN)
        self.play(Write(note))
        self.wait(2)
        self.play(FadeOut(note), FadeOut(VGroup(
            card_tensor, card_label, action_tensor, action_label
        )))


class Figure2Architecture(ThreeDScene):
    """Animation for Figure 2: Architecture Overview of AlphaHold'em."""
    
    def setup(self):
        """Setup common elements and camera configuration."""
        # Set background color
        self.camera.background_color = COLOR_SCHEME["background"]
        
        # Configure camera for 3D scene
        self.set_camera_orientation(phi=25 * DEGREES, theta=-90 * DEGREES, zoom=0.7)
        self.begin_ambient_camera_rotation(rate=0.001)  # Subtle camera rotation
    
    def construct(self):
        """Construct the animation for Figure 2."""
        self.setup()
        
        # Title
        title = Text("AlphaHold'em Architecture", font_size=36)
        title.to_edge(UP)
        title.set_color(COLOR_SCHEME["text"])
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title))
        
        # Create input tensors
        # Card info tensor: s_card ∈ ℝ^{17×4×52}
        card_tensor = create_tensor_3d((0.7, 0.4, 1.2), COLOR_SCHEME["card_tensor"])
        card_tensor.move_to(LEFT * 7)
        card_tensor_label = MathTex(r"s_{card} \in \mathbb{R}^{17 \times 4 \times 52}", color=COLOR_SCHEME["text"]).scale(TEXT_SCALE)
        card_tensor_label.next_to(card_tensor, DOWN, buff=0.3)
        card_tensor_group = VGroup(card_tensor, card_tensor_label)
        
        # Action info tensor: s_action ∈ ℝ^{k×m}
        action_tensor = create_tensor_3d((0.7, 0.4, 1.2), COLOR_SCHEME["action_tensor"])
        action_tensor.move_to(LEFT * 7 + DOWN * 3)
        action_tensor_label = MathTex(r"s_{action} \in \mathbb{R}^{k \times m}", color=COLOR_SCHEME["text"]).scale(TEXT_SCALE)
        action_tensor_label.next_to(action_tensor, DOWN, buff=0.3)
        action_tensor_group = VGroup(action_tensor, action_tensor_label)
        
        # Create ConvNet blocks
        card_convnet = create_network_block(1.5, 1.2, 0.8, "Card ConvNet", COLOR_SCHEME["card_convnet"])
        card_convnet.move_to(LEFT * 4 + UP * 0.5)
        
        action_convnet = create_network_block(1.5, 1.2, 0.8, "Action ConvNet", COLOR_SCHEME["action_convnet"])
        action_convnet.move_to(LEFT * 4 + DOWN * 2.5)
        
        # Create embeddings
        card_embedding = create_tensor_3d((0.8, 0.3, 0.3), COLOR_SCHEME["embedding"])
        card_embedding.move_to(LEFT * 0.5 + UP * 0.5)
        card_embedding_label = Text("Card Embedding", font_size=20).scale(TEXT_SCALE)
        card_embedding_label.set_color(COLOR_SCHEME["text"])
        card_embedding_label.next_to(card_embedding, DOWN, buff=0.2)
        card_embedding_group = VGroup(card_embedding, card_embedding_label)
        
        action_embedding = create_tensor_3d((0.8, 0.3, 0.3), COLOR_SCHEME["embedding"])
        action_embedding.move_to(LEFT * 0.5 + DOWN * 2.5)
        action_embedding_label = Text("Action Embedding", font_size=20).scale(TEXT_SCALE)
        action_embedding_label.set_color(COLOR_SCHEME["text"])
        action_embedding_label.next_to(action_embedding, DOWN, buff=0.2)
        action_embedding_group = VGroup(action_embedding, action_embedding_label)
        
        # Create concatenated embedding
        concat_embedding = create_tensor_3d((1.6, 0.3, 0.3), COLOR_SCHEME["embedding"])
        concat_embedding.move_to(RIGHT * 1.5)
        concat_label = Text("Concatenated\nEmbedding", font_size=20).scale(TEXT_SCALE)
        concat_label.set_color(COLOR_SCHEME["text"])
        concat_label.next_to(concat_embedding, DOWN, buff=0.2)
        concat_group = VGroup(concat_embedding, concat_label)
        
        # Create fully connected layers
        fc_layers = create_network_block(1.5, 1.2, 0.8, "Fully Connected Layers", COLOR_SCHEME["fc_layers"])
        fc_layers.move_to(RIGHT * 4.5)
        
        # Create output heads
        actor_head = create_tensor_3d((0.8, 0.3, 0.3), COLOR_SCHEME["actor_head"])
        actor_head.move_to(RIGHT * 6.5 + UP * 0.8)
        actor_label = MathTex(r"\pi(s)", color=COLOR_SCHEME["text"]).scale(TEXT_SCALE)
        actor_label.next_to(actor_head, RIGHT, buff=0.2)
        actor_desc = Text("Actor Head", font_size=20).scale(TEXT_SCALE)
        actor_desc.set_color(COLOR_SCHEME["text"])
        actor_desc.next_to(actor_head, DOWN, buff=0.2)
        actor_group = VGroup(actor_head, actor_label, actor_desc)
        
        critic_head = create_tensor_3d((0.8, 0.3, 0.3), COLOR_SCHEME["critic_head"])
        critic_head.move_to(RIGHT * 6.5 + DOWN * 0.8)
        critic_label = MathTex(r"V(s)", color=COLOR_SCHEME["text"]).scale(TEXT_SCALE)
        critic_label.next_to(critic_head, RIGHT, buff=0.2)
        critic_desc = Text("Critic Head", font_size=20).scale(TEXT_SCALE)
        critic_desc.set_color(COLOR_SCHEME["text"])
        critic_desc.next_to(critic_head, DOWN, buff=0.2)
        critic_group = VGroup(critic_head, critic_label, critic_desc)
        
        # Create flow arrows
        card_to_convnet = create_flow_arrow(card_tensor, card_convnet, color=COLOR_SCHEME["card_tensor"])
        action_to_convnet = create_flow_arrow(action_tensor, action_convnet, color=COLOR_SCHEME["action_tensor"])
        card_convnet_to_embedding = create_flow_arrow(card_convnet, card_embedding, color=COLOR_SCHEME["card_convnet"])
        action_convnet_to_embedding = create_flow_arrow(action_convnet, action_embedding, color=COLOR_SCHEME["action_convnet"])
        
        # Create vertical arrows for concatenation
        card_to_concat = Arrow(
            start=card_embedding.get_center() + DOWN * (card_embedding.get_height() / 2 + 0.1),
            end=concat_embedding.get_center() + UP * (concat_embedding.get_height() / 2 + 0.1),
            buff=0.1,
            stroke_width=ARROW_STROKE_WIDTH,
            color=COLOR_SCHEME["embedding"]
        )
        
        action_to_concat = Arrow(
            start=action_embedding.get_center() + UP * (action_embedding.get_height() / 2 + 0.1),
            end=concat_embedding.get_center() + DOWN * (concat_embedding.get_height() / 2 + 0.1),
            buff=0.1,
            stroke_width=ARROW_STROKE_WIDTH,
            color=COLOR_SCHEME["embedding"]
        )
        
        concat_to_fc = create_flow_arrow(concat_embedding, fc_layers, color=COLOR_SCHEME["embedding"])
        
        # Create arrows to output heads
        fc_to_actor = Arrow(
            start=fc_layers.get_center() + RIGHT * (fc_layers.get_width() / 2 + 0.1) + UP * 0.4,
            end=actor_head.get_center() + LEFT * (actor_head.get_width() / 2 + 0.1),
            buff=0.1,
            stroke_width=ARROW_STROKE_WIDTH,
            color=COLOR_SCHEME["fc_layers"]
        )
        
        fc_to_critic = Arrow(
            start=fc_layers.get_center() + RIGHT * (fc_layers.get_width() / 2 + 0.1) + DOWN * 0.4,
            end=critic_head.get_center() + LEFT * (critic_head.get_width() / 2 + 0.1),
            buff=0.1,
            stroke_width=ARROW_STROKE_WIDTH,
            color=COLOR_SCHEME["fc_layers"]
        )
        
        # Animation sequence
        # 1. Show input tensors
        self.play(
            FadeIn(card_tensor_group, shift=UP),
            FadeIn(action_tensor_group, shift=UP),
            run_time=ANIMATION_RUN_TIME
        )
        self.wait(0.5)
        
        # 2. Show ConvNet blocks and flow to them
        self.play(
            FadeIn(card_convnet),
            FadeIn(action_convnet),
            Create(card_to_convnet),
            Create(action_to_convnet),
            run_time=ANIMATION_RUN_TIME
        )
        self.wait(0.5)
        
        # 3. Animate signal flow through arrows
        self.play(
            create_signal_flow(card_to_convnet, run_time=1.5),
            create_signal_flow(action_to_convnet, run_time=1.5)
        )
        self.wait(0.5)
        
        # 4. Show embeddings and flow to them
        self.play(
            FadeIn(card_embedding_group),
            FadeIn(action_embedding_group),
            Create(card_convnet_to_embedding),
            Create(action_convnet_to_embedding),
            run_time=ANIMATION_RUN_TIME
        )
        self.wait(0.5)
        
        # 5. Animate signal flow to embeddings
        self.play(
            create_signal_flow(card_convnet_to_embedding, run_time=1.5),
            create_signal_flow(action_convnet_to_embedding, run_time=1.5)
        )
        self.wait(0.5)
        
        # 6. Show concatenation
        self.play(
            FadeIn(concat_group),
            Create(card_to_concat),
            Create(action_to_concat),
            run_time=ANIMATION_RUN_TIME
        )
        self.wait(0.5)
        
        # 7. Animate signal flow to concatenation
        self.play(
            create_signal_flow(card_to_concat, run_time=1.5),
            create_signal_flow(action_to_concat, run_time=1.5)
        )
        self.wait(0.5)
        
        # 8. Show fully connected layers
        self.play(
            FadeIn(fc_layers),
            Create(concat_to_fc),
            run_time=ANIMATION_RUN_TIME
        )
        self.wait(0.5)
        
        # 9. Animate signal flow to fully connected layers
        self.play(
            create_signal_flow(concat_to_fc, run_time=1.5)
        )
        self.wait(0.5)
        
        # 10. Show output heads
        self.play(
            FadeIn(actor_group),
            FadeIn(critic_group),
            Create(fc_to_actor),
            Create(fc_to_critic),
            run_time=ANIMATION_RUN_TIME
        )
        self.wait(0.5)
        
        # 11. Animate signal flow to output heads
        self.play(
            create_signal_flow(fc_to_actor, run_time=1.5),
            create_signal_flow(fc_to_critic, run_time=1.5)
        )
        self.wait(0.5)
        
        # 12. Final camera rotation for better view
        self.move_camera(phi=65 * DEGREES, theta=-40 * DEGREES, run_time=2)
        self.wait(1)
        
        # Fade out title for transition to next figure
        self.play(FadeOut(title))
        self.wait(1)


class Figure3Losses(ThreeDScene):
    """Animation for Figure 3: Losses and Backpropagation in AlphaHold'em."""
    
    def setup(self):
        """Setup common elements and camera configuration."""
        # Set background color
        self.camera.background_color = COLOR_SCHEME["background"]
        
        # Configure camera for 3D scene
        self.set_camera_orientation(phi=25 * DEGREES, theta=-90 * DEGREES)
        self.begin_ambient_camera_rotation(rate=0.005)  # Subtle camera rotation
    
    def construct(self):
        """Construct the animation for Figure 3."""
        self.setup()
        
        # Title
        title = Text("AlphaHold'em Losses & Backpropagation", font_size=36)
        title.to_edge(UP)
        title.set_color(COLOR_SCHEME["text"])
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title))
        
        # Create simplified architecture (reusing elements from Figure 2)
        # We'll create a more compact version focusing on the loss computation
        
        # Create network block
        network_block = create_network_block(2.5, 2.0, 1.0, "AlphaHold'em Network", COLOR_SCHEME["fc_layers"])
        network_block.move_to(LEFT * 3)
        
        # Create output heads
        actor_head = create_tensor_3d((0.8, 0.3, 0.3), COLOR_SCHEME["actor_head"])
        actor_head.move_to(LEFT * 0.5 + UP * 1)
        actor_label = MathTex(r"\pi(s)", color=COLOR_SCHEME["text"]).scale(TEXT_SCALE)
        actor_label.next_to(actor_head, RIGHT, buff=0.2)
        actor_desc = Text("Actor Head", font_size=20).scale(TEXT_SCALE)
        actor_desc.set_color(COLOR_SCHEME["text"])
        actor_desc.next_to(actor_head, DOWN, buff=0.2)
        actor_group = VGroup(actor_head, actor_label, actor_desc)
        
        critic_head = create_tensor_3d((0.8, 0.3, 0.3), COLOR_SCHEME["critic_head"])
        critic_head.move_to(LEFT * 0.5 + DOWN * 1)
        critic_label = MathTex(r"V(s)", color=COLOR_SCHEME["text"]).scale(TEXT_SCALE)
        critic_label.next_to(critic_head, RIGHT, buff=0.2)
        critic_desc = Text("Critic Head", font_size=20).scale(TEXT_SCALE)
        critic_desc.set_color(COLOR_SCHEME["text"])
        critic_desc.next_to(critic_head, DOWN, buff=0.2)
        critic_group = VGroup(critic_head, critic_label, critic_desc)
        
        # Create arrows from network to heads
        network_to_actor = create_flow_arrow(network_block, actor_head, color=COLOR_SCHEME["fc_layers"])
        network_to_critic = create_flow_arrow(network_block, critic_head, color=COLOR_SCHEME["fc_layers"])
        
        # Create loss computation blocks
        action_loss_block = create_tensor_3d((1.0, 1.0, 0.5), COLOR_SCHEME["loss_action"])
        action_loss_block.move_to(RIGHT * 2 + UP * 1)
        action_loss_label = MathTex(r"-\log \pi(a|s)", color=COLOR_SCHEME["text"]).scale(TEXT_SCALE)
        action_loss_label.next_to(action_loss_block, RIGHT, buff=0.2)
        action_loss_desc = Text("Action Loss", font_size=20).scale(TEXT_SCALE)
        action_loss_desc.set_color(COLOR_SCHEME["text"])
        action_loss_desc.next_to(action_loss_block, DOWN, buff=0.2)
        action_loss_group = VGroup(action_loss_block, action_loss_label, action_loss_desc)
        
        value_loss_block = create_tensor_3d((1.0, 1.0, 0.5), COLOR_SCHEME["loss_value"])
        value_loss_block.move_to(RIGHT * 2 + DOWN * 1)
        value_loss_label = MathTex(r"(r + \gamma V(s') - V(s))^2", color=COLOR_SCHEME["text"]).scale(TEXT_SCALE)
        value_loss_label.next_to(value_loss_block, RIGHT, buff=0.2)
        value_loss_desc = Text("Value Loss", font_size=20).scale(TEXT_SCALE)
        value_loss_desc.set_color(COLOR_SCHEME["text"])
        value_loss_desc.next_to(value_loss_block, DOWN, buff=0.2)
        value_loss_group = VGroup(value_loss_block, value_loss_label, value_loss_desc)
        
        # Create arrows to loss blocks
        actor_to_loss = create_flow_arrow(actor_head, action_loss_block, color=COLOR_SCHEME["actor_head"])
        critic_to_loss = create_flow_arrow(critic_head, value_loss_block, color=COLOR_SCHEME["critic_head"])
        
        # Create combined loss
        combined_loss_block = create_tensor_3d((1.0, 1.0, 0.5), COLOR_SCHEME["gradient"])
        combined_loss_block.move_to(RIGHT * 4.5)
        combined_loss_label = MathTex(r"L_{total} = L_{actor} + L_{critic}", color=COLOR_SCHEME["text"]).scale(TEXT_SCALE)
        combined_loss_label.next_to(combined_loss_block, RIGHT, buff=0.2)
        combined_loss_desc = Text("Total Loss", font_size=20).scale(TEXT_SCALE)
        combined_loss_desc.set_color(COLOR_SCHEME["text"])
        combined_loss_desc.next_to(combined_loss_block, DOWN, buff=0.2)
        combined_loss_group = VGroup(combined_loss_block, combined_loss_label, combined_loss_desc)
        
        # Create arrows to combined loss
        action_loss_to_combined = Arrow(
            start=action_loss_block.get_center() + RIGHT * (action_loss_block.get_width() / 2 + 0.1) + DOWN * 0.2,
            end=combined_loss_block.get_center() + LEFT * (combined_loss_block.get_width() / 2 + 0.1) + UP * 0.2,
            buff=0.1,
            stroke_width=ARROW_STROKE_WIDTH,
            color=COLOR_SCHEME["loss_action"]
        )
        
        value_loss_to_combined = Arrow(
            start=value_loss_block.get_center() + RIGHT * (value_loss_block.get_width() / 2 + 0.1) + UP * 0.2,
            end=combined_loss_block.get_center() + LEFT * (combined_loss_block.get_width() / 2 + 0.1) + DOWN * 0.2,
            buff=0.1,
            stroke_width=ARROW_STROKE_WIDTH,
            color=COLOR_SCHEME["loss_value"]
        )
        
        # Create backpropagation arrows (gradient flow)
        backprop_arrow = Arrow(
            start=combined_loss_block.get_center() + LEFT * (combined_loss_block.get_width() / 2 + 0.1),
            end=network_block.get_center() + RIGHT * (network_block.get_width() / 2 + 0.1),
            buff=0.1,
            stroke_width=ARROW_STROKE_WIDTH * 1.5,
            color=COLOR_SCHEME["gradient"]
        )
        backprop_label = Text("Backpropagation", font_size=24).scale(TEXT_SCALE)
        backprop_label.set_color(COLOR_SCHEME["gradient"])
        backprop_label.next_to(backprop_arrow, UP, buff=0.2)
        
        # TD Error label for critic
        td_error_label = Text("TD Error", font_size=20).scale(TEXT_SCALE)
        td_error_label.set_color(COLOR_SCHEME["text"])
        td_error_label.next_to(critic_to_loss, UP, buff=0.2)
        
        # Policy Gradient label for actor
        policy_gradient_label = Text("Policy Gradient", font_size=20).scale(TEXT_SCALE)
        policy_gradient_label.set_color(COLOR_SCHEME["text"])
        policy_gradient_label.next_to(actor_to_loss, UP, buff=0.2)
        
        # Animation sequence
        # 1. Show network and output heads
        self.play(
            FadeIn(network_block),
            run_time=ANIMATION_RUN_TIME
        )
        self.wait(0.5)
        
        self.play(
            FadeIn(actor_group),
            FadeIn(critic_group),
            Create(network_to_actor),
            Create(network_to_critic),
            run_time=ANIMATION_RUN_TIME
        )
        self.wait(0.5)
        
        # 2. Animate signal flow to output heads
        self.play(
            create_signal_flow(network_to_actor, run_time=1.5),
            create_signal_flow(network_to_critic, run_time=1.5)
        )
        self.wait(0.5)
        
        # 3. Show loss computation blocks
        self.play(
            FadeIn(action_loss_group),
            FadeIn(value_loss_group),
            Create(actor_to_loss),
            Create(critic_to_loss),
            run_time=ANIMATION_RUN_TIME
        )
        self.wait(0.5)
        
        # 4. Show TD Error and Policy Gradient labels
        self.play(
            FadeIn(td_error_label),
            FadeIn(policy_gradient_label),
            run_time=ANIMATION_RUN_TIME
        )
        self.wait(0.5)
        
        # 5. Animate signal flow to loss blocks
        self.play(
            create_signal_flow(actor_to_loss, run_time=1.5),
            create_signal_flow(critic_to_loss, run_time=1.5)
        )
        self.wait(0.5)
        
        # 6. Show combined loss
        self.play(
            FadeIn(combined_loss_group),
            Create(action_loss_to_combined),
            Create(value_loss_to_combined),
            run_time=ANIMATION_RUN_TIME
        )
        self.wait(0.5)
        
        # 7. Animate signal flow to combined loss
        self.play(
            create_signal_flow(action_loss_to_combined, run_time=1.5),
            create_signal_flow(value_loss_to_combined, run_time=1.5)
        )
        self.wait(0.5)
        
        # 8. Show backpropagation
        self.play(
            Create(backprop_arrow),
            Write(backprop_label),
            run_time=ANIMATION_RUN_TIME
        )
        self.wait(0.5)
        
        # 9. Animate gradient flow (backpropagation)
        # Create multiple gradient dots for a more dramatic effect
        gradient_dots = VGroup()
        for _ in range(5):
            dot = Dot(color=COLOR_SCHEME["gradient"], radius=0.05)
            dot.move_to(combined_loss_block.get_center())
            gradient_dots.add(dot)
            
        self.play(FadeIn(gradient_dots))
        
        # Animate dots flowing backward
        for dot in gradient_dots:
            self.play(
                MoveAlongPath(dot, backprop_arrow, run_time=1.5),
                rate_func=linear
            )
            
        self.play(FadeOut(gradient_dots))
        self.wait(0.5)
        
        # 10. Final camera rotation for better view
        self.move_camera(phi=70 * DEGREES, theta=-35 * DEGREES, run_time=2)
        self.wait(1)
        
        # Fade out title for transition to next figure
        self.play(FadeOut(title))
        self.wait(1)


class Figure4TrainingPipeline(ThreeDScene):
    """Animation for Figure 4: Training Pipeline of AlphaHold'em."""
    
    def setup(self):
        """Setup common elements and camera configuration."""
        # Set background color
        self.camera.background_color = COLOR_SCHEME["background"]
        
        # Configure camera for 3D scene
        self.set_camera_orientation(phi=25 * DEGREES, theta=-90 * DEGREES)
        self.begin_ambient_camera_rotation(rate=0.005)  # Subtle camera rotation
    
    def construct(self):
        """Construct the animation for Figure 4."""
        self.setup()
        
        # Title
        title = Text("AlphaHold'em Training Pipeline", font_size=36)
        title.to_edge(UP)
        title.set_color(COLOR_SCHEME["text"])
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title))
        
        # Create environment block
        environment = create_network_block(2.0, 1.5, 1.0, "Poker Environment", COLOR_SCHEME["environment"])
        environment.move_to(LEFT * 5)
        
        # Create experience tuple
        experience_tuple = create_tensor_3d((0.8, 1.5, 0.5), COLOR_SCHEME["environment"])
        experience_tuple.move_to(LEFT * 2.5)
        experience_label = MathTex(r"(s, a, r, s')", color=COLOR_SCHEME["text"]).scale(TEXT_SCALE)
        experience_label.next_to(experience_tuple, DOWN, buff=0.2)
        experience_group = VGroup(experience_tuple, experience_label)
        
        # Create replay buffer
        replay_buffer = create_tensor_3d((2.0, 1.5, 1.0), COLOR_SCHEME["replay_buffer"])
        replay_buffer.move_to(LEFT * 0)
        replay_buffer_label = Text("Replay Buffer", font_size=24).scale(TEXT_SCALE)
        replay_buffer_label.set_color(COLOR_SCHEME["text"])
        replay_buffer_label.next_to(replay_buffer, DOWN, buff=0.2)
        replay_buffer_group = VGroup(replay_buffer, replay_buffer_label)
        
        # Create mini-batch
        mini_batch = create_tensor_3d((0.8, 1.0, 0.5), COLOR_SCHEME["replay_buffer"])
        mini_batch.move_to(RIGHT * 2.5)
        mini_batch_label = Text("Mini-batch", font_size=20).scale(TEXT_SCALE)
        mini_batch_label.set_color(COLOR_SCHEME["text"])
        mini_batch_label.next_to(mini_batch, DOWN, buff=0.2)
        mini_batch_group = VGroup(mini_batch, mini_batch_label)
        
        # Create network block (simplified)
        network_block = create_network_block(2.0, 1.5, 1.0, "AlphaHold'em Network", COLOR_SCHEME["fc_layers"])
        network_block.move_to(RIGHT * 5)
        
        # Create flow arrows
        env_to_exp = create_flow_arrow(environment, experience_tuple, color=COLOR_SCHEME["environment"])
        exp_to_buffer = create_flow_arrow(experience_tuple, replay_buffer, color=COLOR_SCHEME["environment"])
        buffer_to_batch = create_flow_arrow(replay_buffer, mini_batch, color=COLOR_SCHEME["replay_buffer"])
        batch_to_network = create_flow_arrow(mini_batch, network_block, color=COLOR_SCHEME["replay_buffer"])
        
        # Create update arrow (feedback loop)
        update_arrow = Arrow(
            start=network_block.get_center() + UP * (network_block.get_height() / 2 + 0.1),
            end=environment.get_center() + UP * (environment.get_height() / 2 + 0.1),
            path_arc=-1.5,
            buff=0.1,
            stroke_width=ARROW_STROKE_WIDTH,
            color=COLOR_SCHEME["gradient"]
        )
        update_label = Text("Policy Update", font_size=24).scale(TEXT_SCALE)
        update_label.set_color(COLOR_SCHEME["gradient"])
        update_label.next_to(update_arrow, UP, buff=0.2)
        
        # Create step labels
        step1_label = Text("1. Generate Experience", font_size=20).scale(TEXT_SCALE)
        step1_label.set_color(COLOR_SCHEME["text"])
        step1_label.next_to(env_to_exp, DOWN, buff=0.5)
        
        step2_label = Text("2. Store in Buffer", font_size=20).scale(TEXT_SCALE)
        step2_label.set_color(COLOR_SCHEME["text"])
        step2_label.next_to(exp_to_buffer, DOWN, buff=0.5)
        
        step3_label = Text("3. Sample Mini-batch", font_size=20).scale(TEXT_SCALE)
        step3_label.set_color(COLOR_SCHEME["text"])
        step3_label.next_to(buffer_to_batch, DOWN, buff=0.5)
        
        step4_label = Text("4. Train Network", font_size=20).scale(TEXT_SCALE)
        step4_label.set_color(COLOR_SCHEME["text"])
        step4_label.next_to(batch_to_network, DOWN, buff=0.5)
        
        step5_label = Text("5. Update Policy", font_size=20).scale(TEXT_SCALE)
        step5_label.set_color(COLOR_SCHEME["text"])
        step5_label.next_to(update_arrow, RIGHT, buff=0.5)
        
        # Animation sequence
        # 1. Show environment
        self.play(
            FadeIn(environment),
            run_time=ANIMATION_RUN_TIME
        )
        self.wait(0.5)
        
        # 2. Generate experience
        self.play(
            FadeIn(experience_group),
            Create(env_to_exp),
            FadeIn(step1_label),
            run_time=ANIMATION_RUN_TIME
        )
        self.wait(0.5)
        
        # 3. Animate signal flow for experience generation
        self.play(
            create_signal_flow(env_to_exp, run_time=1.5)
        )
        self.wait(0.5)
        
        # 4. Store in buffer
        self.play(
            FadeIn(replay_buffer_group),
            Create(exp_to_buffer),
            FadeIn(step2_label),
            run_time=ANIMATION_RUN_TIME
        )
        self.wait(0.5)
        
        # 5. Animate signal flow for storing in buffer
        self.play(
            create_signal_flow(exp_to_buffer, run_time=1.5)
        )
        self.wait(0.5)
        
        # 6. Sample mini-batch
        self.play(
            FadeIn(mini_batch_group),
            Create(buffer_to_batch),
            FadeIn(step3_label),
            run_time=ANIMATION_RUN_TIME
        )
        self.wait(0.5)
        
        # 7. Animate signal flow for sampling mini-batch
        self.play(
            create_signal_flow(buffer_to_batch, run_time=1.5)
        )
        self.wait(0.5)
        
        # 8. Train network
        self.play(
            FadeIn(network_block),
            Create(batch_to_network),
            FadeIn(step4_label),
            run_time=ANIMATION_RUN_TIME
        )
        self.wait(0.5)
        
        # 9. Animate signal flow for training
        self.play(
            create_signal_flow(batch_to_network, run_time=1.5)
        )
        self.wait(0.5)
        
        # 10. Update policy (feedback loop)
        self.play(
            Create(update_arrow),
            Write(update_label),
            FadeIn(step5_label),
            run_time=ANIMATION_RUN_TIME
        )
        self.wait(0.5)
        
        # 11. Animate signal flow for policy update
        self.play(
            create_signal_flow(update_arrow, run_time=2.0)
        )
        self.wait(0.5)
        
        # 12. Highlight the complete cycle
        cycle_arrows = VGroup(env_to_exp, exp_to_buffer, buffer_to_batch, batch_to_network, update_arrow)
        self.play(
            cycle_arrows.animate.set_stroke(width=ARROW_STROKE_WIDTH * 1.5),
            run_time=ANIMATION_RUN_TIME
        )
        self.wait(0.5)
        
        # 13. Animate full cycle flow
        self.play(
            create_signal_flow(env_to_exp, run_time=1.0),
            create_signal_flow(exp_to_buffer, run_time=1.0),
            create_signal_flow(buffer_to_batch, run_time=1.0),
            create_signal_flow(batch_to_network, run_time=1.0),
            create_signal_flow(update_arrow, run_time=1.0),
            lag_ratio=0.2
        )
        self.wait(0.5)
        
        # 14. Final camera rotation for better view
        self.move_camera(phi=60 * DEGREES, theta=-45 * DEGREES, run_time=2)
        self.wait(1)
        
        # 15. Show final title
        final_title = Text("AlphaHold'em Complete Training Pipeline", font_size=36)
        final_title.to_edge(UP)
        final_title.set_color(COLOR_SCHEME["text"])
        self.add_fixed_in_frame_mobjects(final_title)
        self.play(
            FadeOut(title),
            FadeIn(final_title)
        )
        self.wait(2)


class AlphaHoldemAnimation(Scene):
    """Main scene that combines all three figures into a complete animation."""
    
    def construct(self):
        """Construct the complete animation."""
        # Title
        title = Text("AlphaHold'em: Deep Reinforcement Learning for Poker", font_size=48)
        title.to_edge(UP)
        
        # Subtitle
        subtitle = Text("Visualizing Architecture, Losses, and Training Pipeline", font_size=36)
        subtitle.next_to(title, DOWN)
        
        # Show title and subtitle
        self.play(Write(title))
        self.play(Write(subtitle))
        self.wait(1)
        
        # Fade out title and subtitle
        self.play(
            FadeOut(title),
            FadeOut(subtitle)
        )
        
        # Display text for Figure 2
        fig2_text = Text("Figure 2: Architecture Overview", font_size=36)
        fig2_text.to_edge(UP)
        self.play(Write(fig2_text))
        self.wait(1)
        self.play(FadeOut(fig2_text))
        
        # Note: In a real Manim script, we would use the following to include Figure2Architecture
        self.play(FadeIn(Figure2Architecture().mobjects))
        # However, for this script, we'll just indicate where it would be included
        
        # Display text for Figure 3
        fig3_text = Text("Figure 3: Losses and Backpropagation", font_size=36)
        fig3_text.to_edge(UP)
        self.play(Write(fig3_text))
        self.wait(1)
        self.play(FadeOut(fig3_text))
        
        # Note: In a real Manim script, we would include Figure3Losses here
        self.play(FadeIn(Figure3Losses().mobjects))
        
        # Display text for Figure 4
        fig4_text = Text("Figure 4: Training Pipeline", font_size=36)
        fig4_text.to_edge(UP)
        self.play(Write(fig4_text))
        self.wait(1)
        self.play(FadeOut(fig4_text))
        
        # Note: In a real Manim script, we would include Figure4TrainingPipeline here
        self.play(FadeIn(Figure4TrainingPipeline().mobjects))

        # Final title
        final_title = Text("AlphaHold'em: Complete Visualization", font_size=48)
        final_title.to_edge(UP)
        self.play(Write(final_title))
        self.wait(2)
        self.play(FadeOut(final_title))


if __name__ == "__main__":
    # Uncomment the scene you want to render
    # Command to render: manim -pql alphaholdem_animation.py Figure2Architecture
    
    # For Figure 2 (Architecture Overview)
    # python3 -m manim -pql --watch alphaholdem_animation.py Figure2Architecture
    
    # For Figure 3 (Losses)
    # python3 -m manim -pql alphaholdem_animation.py Figure3Losses
    
    # For Figure 4 (Training Pipeline)
    # python3 -m manim -pql alphaholdem_animation.py Figure4TrainingPipeline
    
    # For the complete animation
    # python3 -m manim -pql alphaholdem_animation.py AlphaHoldemAnimation

    #manim -pql --watch actor_critic.py ActorCriticDiagram

    pass
