from manim import *

class ActorCriticDiagram(ThreeDScene):
    def construct(self):
        # Slight 3D camera tilt (optional perspective effect)
        self.set_camera_orientation(phi=25 * DEGREES, theta=-90 * DEGREES)

        # Card Information Input
        card_input = Cube(side_length=1, fill_color=BLUE, fill_opacity=0.5)
        card_label = Text("Card Info", font_size=24).next_to(card_input, DOWN)
        card_group = VGroup(card_input, card_label).shift(LEFT * 4)

        # Action Information Input
        action_input = Cube(side_length=1, fill_color=GREEN, fill_opacity=0.5)
        action_label = Text("Action Info", font_size=24).next_to(action_input, DOWN)
        action_group = VGroup(action_input, action_label).shift(LEFT * 4 + DOWN * 2)

        # Card ConvNet
        card_conv = Rectangle(width=1.5, height=1, fill_color=BLUE_E, fill_opacity=0.5)
        card_conv_label = Text("Card ConvNet", font_size=24).next_to(card_conv, UP)
        card_conv_group = VGroup(card_conv, card_conv_label).next_to(card_group, RIGHT, buff=1)

        # Action ConvNet
        action_conv = Rectangle(width=1.5, height=1, fill_color=GREEN_E, fill_opacity=0.5)
        action_conv_label = Text("Action ConvNet", font_size=24).next_to(action_conv, UP)
        action_conv_group = VGroup(action_conv, action_conv_label).next_to(action_group, RIGHT, buff=1)

        # Fully Connected Layers
        fc_layer = Rectangle(width=2, height=1, fill_color=YELLOW, fill_opacity=0.5)
        fc_label = Text("Fully Connected", font_size=24).next_to(fc_layer, UP)
        fc_group = VGroup(fc_layer, fc_label).shift(RIGHT * 2)

        # Output Layer
        output = Rectangle(width=1.5, height=1, fill_color=RED, fill_opacity=0.5)
        output_label = Text("Action Probabilities", font_size=24).next_to(output, UP)
        output_group = VGroup(output, output_label).next_to(fc_group, RIGHT, buff=1)

        # Arrows
        arrows = VGroup(
            Arrow(card_group.get_right(), card_conv_group.get_left(), buff=0.1),
            Arrow(action_group.get_right(), action_conv_group.get_left(), buff=0.1),
            Arrow(card_conv_group.get_right(), fc_group.get_left(), buff=0.1),
            Arrow(action_conv_group.get_right(), fc_group.get_left(), buff=0.1),
            Arrow(fc_group.get_right(), output_group.get_left(), buff=0.1)
        )

        # Assemble all components
        architecture = VGroup(
            card_group,
            action_group,
            card_conv_group,
            action_conv_group,
            fc_group,
            output_group,
            arrows
        )

        # Add to scene
        self.play(FadeIn(architecture))
        self.wait(2)

        # # Stacked Tensor Representation (17x4x52) as overlapping slices
        # tensor_slices = VGroup()
        # for i in range(6):  # Fewer than 17 for visual clarity
        #     slice_rect = Rectangle(width=1.5, height=0.6,
        #                            fill_color=PURPLE,
        #                            fill_opacity=0.4,
        #                            stroke_opacity=0.2)
        #     # Add slight OUT offset to simulate depth
        #     slice_rect.shift(LEFT * 6 + OUT * (i * 0.1))
        #     tensor_slices.add(slice_rect)

        # tensor_label = MathTex(r"\textbf{s} \in \mathbb{R}^{17 \times 4 \times 52}").scale(0.5).next_to(tensor_slices, DOWN)

        # # State node
        # state = Circle(radius=0.3, color=WHITE).shift(LEFT * 4)
        # state_label = Text("State", font_size=24).next_to(state, UP)
        # tensor_to_state = Arrow(start=tensor_slices[0].get_right(), end=state.get_left(), buff=0.1)

        # # Actor network
        # actor_layers = VGroup(
        #     Circle(radius=0.3, color=BLUE),
        #     Circle(radius=0.3, color=BLUE),
        #     Circle(radius=0.3, color=BLUE),
        # ).arrange(DOWN, buff=0.6).shift(LEFT * 2)
        # actor_label = Text("Actor Network", font_size=24).next_to(actor_layers, UP)

        # # Action output
        # action = Circle(radius=0.3, color=GREEN).shift(RIGHT)
        # action_label = Text("π(s)", font_size=24).next_to(action, DOWN)

        # # Critic network
        # critic_layers = VGroup(
        #     Circle(radius=0.3, color=RED),
        #     Circle(radius=0.3, color=RED),
        #     Circle(radius=0.3, color=RED),
        # ).arrange(DOWN, buff=0.6).shift(RIGHT * 3)
        # critic_label = Text("Critic Network", font_size=24).next_to(critic_layers, UP)

        # # Value output
        # value = Circle(radius=0.3, color=YELLOW).shift(RIGHT * 5)
        # value_label = Text("Q(s, a)", font_size=24).next_to(value, DOWN)

        # # Forward arrows
        # input_arrow = Arrow(start=state.get_right(), end=actor_layers[1].get_left(), buff=0.1)
        # actor_to_action = Arrow(start=actor_layers[1].get_right(), end=action.get_left(), buff=0.1)
        # action_to_critic = Arrow(start=action.get_right(), end=critic_layers[1].get_left(), buff=0.1)
        # critic_to_value = Arrow(start=critic_layers[1].get_right(), end=value.get_left(), buff=0.1)

        # arrows_forward = [input_arrow, actor_to_action, action_to_critic, critic_to_value]

        # # Backward arrows (for gradients)
        # grad1 = Arrow(start=value.get_left(), end=critic_layers[1].get_right(), color=ORANGE, buff=0.1)
        # grad2 = Arrow(start=critic_layers[1].get_left(), end=action.get_right(), color=ORANGE, buff=0.1)
        # grad3 = Arrow(start=action.get_left(), end=actor_layers[1].get_right(), color=ORANGE, buff=0.1)
        # arrows_backward = [grad1, grad2, grad3]

        # # Create static elements
        # self.play(Create(tensor_slices), Write(tensor_label))
        # self.play(Create(state), Write(state_label), Create(tensor_to_state))
        # self.play(Create(actor_layers), Write(actor_label))
        # self.play(Create(action), Write(action_label))
        # self.play(Create(critic_layers), Write(critic_label))
        # self.play(Create(value), Write(value_label))
        # self.play(*[Create(a) for a in arrows_forward])
        # self.wait(0.5)

        # # Forward pass animations
        # tensor_dot = Dot(color=PURPLE).move_to(tensor_slices[0].get_right())
        # self.play(MoveAlongPath(tensor_dot, tensor_to_state), run_time=1)

        # dots_forward = [
        #     Dot(color=WHITE).move_to(state.get_right()),
        #     Dot(color=BLUE).move_to(actor_layers[1].get_right()),
        #     Dot(color=GREEN).move_to(action.get_right()),
        #     Dot(color=YELLOW).move_to(critic_layers[1].get_right()),
        # ]

        # for dot, path in zip(dots_forward, arrows_forward):
        #     self.play(MoveAlongPath(dot, path), run_time=1)

        # self.wait(0.5)

        # # Backward pass
        # self.play(*[Create(a) for a in arrows_backward])
        # dots_backward = [
        #     Dot(color=ORANGE).move_to(value.get_left()),
        #     Dot(color=ORANGE).move_to(critic_layers[1].get_left()),
        #     Dot(color=ORANGE).move_to(action.get_left()),
        # ]

        # for dot, path in zip(dots_backward, arrows_backward):
        #     self.play(MoveAlongPath(dot, path), run_time=1)

        # self.wait()
